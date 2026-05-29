"""Tagged-action protocol adapter — multi-turn tool-calling via native function calling.

Handles protocols where the orchestrator (green agent) sends plain text
messages via A2A TextParts describing available tools and expects responses
as tagged JSON actions: <json>{"name": "...", "arguments": {...}}</json>

Instead of prompting the LLM to generate JSON in free text, this adapter:
1. Parses the tool list from the first message into native OpenAI tool defs
2. Uses the model's trained function-calling capability
3. Maps the structured tool_calls response back to <json> format

If the tool list cannot be parsed (non-standard format), falls back to
text-prompting with <json> tag generation.

This gives better tool selection, argument accuracy, and turn efficiency
compared to text-mediated tool calling.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI

from usage import tracker

logger = logging.getLogger("agentwhetters.tagged_action_adapter")

# Detection pattern: the orchestrator includes this instruction
_TAGGED_ACTION_DETECTION_PATTERN = re.compile(
    r"Please wrap the JSON part with <json>", re.IGNORECASE
)

# Also check for tool list format
_TAGGED_ACTION_TOOL_LIST_PATTERN = re.compile(
    r"Here's a list of tools you can use", re.IGNORECASE
)

RESPOND_ACTION_NAME = "respond"

# The "respond" tool definition for direct user replies
_RESPOND_TOOL = {
    "type": "function",
    "function": {
        "name": "respond",
        "description": "Send a direct text reply to the user. Use this when you need to communicate with the user rather than call a tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message to send to the user.",
                }
            },
            "required": ["content"],
        },
    },
}


def is_tagged_action_message(text: str) -> bool:
    """Detect if a message uses the tagged-action protocol (tools in text, <json> responses)."""
    return bool(
        _TAGGED_ACTION_DETECTION_PATTERN.search(text)
        and _TAGGED_ACTION_TOOL_LIST_PATTERN.search(text)
    )


# Backward-compatible alias
is_tau2_protocol_message = is_tagged_action_message


def _make_client() -> AsyncOpenAI:
    """Create an OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    if azure_endpoint:
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        return AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _parse_tools_from_message(text: str) -> list[dict[str, Any]] | None:
    """Extract OpenAI-format tool definitions from the first message.

    The green agent sends tools as a JSON array after "Here's a list of tools..."
    Each element has {"type": "function", "function": {"name": ..., "parameters": ...}}.
    Returns None if parsing fails (triggers text-prompting fallback).
    """
    # Find the JSON array after the tools marker
    match = re.search(
        r"Here's a list of tools you can use[^\n]*:\s*\n(\[.*?\n\])",
        text,
        re.DOTALL,
    )
    if not match:
        return None
    try:
        tools = json.loads(match.group(1))
        if not isinstance(tools, list) or not tools:
            return None
        # Validate first element has expected structure
        if "function" not in tools[0]:
            return None
        return tools
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def _extract_policy_and_user_message(text: str) -> tuple[str, str | None]:
    """Split the first message into policy context and initial user message.

    The first message typically has:
    - Policy/rules text
    - Tool list (which we extract separately)
    - Format instructions (which we skip since we use native tools)
    - Sometimes an initial user query at the end

    Returns (policy_text, user_query_or_none).
    """
    # Remove the tool list and format instructions to get the policy
    # Everything before "Here's a list of tools" is policy
    parts = re.split(r"Here's a list of tools you can use[^\n]*:", text, maxsplit=1)
    policy = parts[0].strip() if parts else text

    # Check for a user message after the format instructions
    # Look for text after the examples block
    user_msg = None
    after_format = re.search(
        r"You cannot respond to user and use a tool at the same time!!\s*(?:Examples.*?```\s*)?(.*)",
        text,
        re.DOTALL,
    )
    if after_format:
        remainder = after_format.group(1).strip()
        # If there's substantial text after format instructions, it's the user message
        if len(remainder) > 20:
            user_msg = remainder

    return policy, user_msg


_SYSTEM_PROMPT = """\
You are a helpful agent. Follow the policy strictly and use the available tools \
to fulfill the user's request.

RULES:
- Use ONE tool at a time.
- Be PROACTIVE: look up information yourself rather than asking the user.
- When you have a user/account ID, immediately call the lookup tool.
- Minimize back-and-forth. Gather data via tool calls BEFORE responding.
- Only ask the user for information you cannot find via tools.
- When policy requires confirmation, ask ONCE with all details, then act.
- Keep responses concise.
- ALWAYS execute write operations (update, cancel, book) after user confirms. \
Do not stop at just reporting information.
"""


class TaggedActionAdapter:
    """Handles multi-turn tagged-action tool-calling protocol using native function calling."""

    def __init__(self):
        self._client = _make_client()
        self._model = os.environ.get("AGENT_MODEL", "gpt-5.4")
        # context_id -> session state
        self._sessions: dict[str, _Session] = {}

    async def handle_turn(self, context_id: str, text: str) -> str:
        """Process a tagged-action turn and return the response text.

        Args:
            context_id: Session key for multi-turn state.
            text: The raw text message from the green agent.

        Returns:
            Response text containing <json>...</json> tags.
        """
        if context_id not in self._sessions:
            # First turn: parse tools and set up session
            session = self._init_session(text)
            self._sessions[context_id] = session
        else:
            session = self._sessions[context_id]
            # Subsequent turns: tool results or user observations
            session.messages.append({"role": "user", "content": text})

        return await self._call_llm(session)

    def _init_session(self, first_message: str) -> "_Session":
        """Initialize a session from the first message."""
        tools = _parse_tools_from_message(first_message)

        if tools is not None:
            # Native function calling mode
            # Add the "respond" tool for direct user replies
            tool_names = {t["function"]["name"] for t in tools}
            if "respond" not in tool_names:
                tools.append(_RESPOND_TOOL)

            policy, user_msg = _extract_policy_and_user_message(first_message)
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT + "\n\nPOLICY:\n" + policy},
            ]
            # If we extracted a user message, add it separately
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            else:
                # Use the full first message as user context
                messages.append({"role": "user", "content": first_message})

            return _Session(messages=messages, tools=tools, native_mode=True)
        else:
            # Fallback: text-prompting mode (no parseable tool list)
            logger.info("Could not parse native tools; falling back to text mode")
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT_FALLBACK},
                {"role": "user", "content": first_message},
            ]
            return _Session(messages=messages, tools=None, native_mode=False)

    async def _call_llm(self, session: "_Session") -> str:
        """Make an LLM call and format the response as <json> tags."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": session.messages,
        }
        if session.native_mode and session.tools:
            kwargs["tools"] = session.tools
            kwargs["tool_choice"] = "required"

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            logger.exception("tagged_action adapter LLM call failed: %s", exc)
            fallback = json.dumps(
                {"name": RESPOND_ACTION_NAME, "arguments": {"content": f"Error: {exc}"}}
            )
            return f"<json>{fallback}</json>"

        tracker.record(response, label="tagged_action")
        message = response.choices[0].message

        if session.native_mode and message.tool_calls:
            # Native mode: convert tool_calls to <json> format
            tool_call = message.tool_calls[0]
            name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {"raw": tool_call.function.arguments}

            # Store assistant message with tool_call in history
            session.messages.append(message.model_dump(exclude_none=True))

            result = json.dumps({"name": name, "arguments": arguments})
            return f"<json>{result}</json>"
        else:
            # Text mode or content-only response
            content = message.content or ""
            session.messages.append({"role": "assistant", "content": content})

            if session.native_mode:
                # Model returned content instead of tool_call; wrap as respond
                result = json.dumps(
                    {"name": RESPOND_ACTION_NAME, "arguments": {"content": content}}
                )
                return f"<json>{result}</json>"

            # Fallback text mode: ensure <json> tags
            if "<json>" not in content:
                return self._fix_response_format(content)
            return self._validate_json_in_tags(content)

    def _validate_json_in_tags(self, content: str) -> str:
        """Validate JSON inside <json> tags and fix common LLM errors."""
        match = re.search(r"<json>(.*?)</json>", content, re.DOTALL)
        if not match:
            return content
        json_str = match.group(1).strip()
        try:
            json.loads(json_str)
            return content
        except json.JSONDecodeError:
            fixed = self._extract_valid_json(json_str)
            if fixed:
                return f"<json>{fixed}</json>"
            return self._fix_response_format(content)

    def _extract_valid_json(self, text: str) -> str | None:
        """Extract a valid JSON object from text using brace counting."""
        start = text.find("{")
        if start < 0:
            return None
        brace_count = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    candidate = text[start:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        return None
        return None

    def _fix_response_format(self, content: str) -> str:
        """Attempt to extract JSON and wrap in <json> tags if model forgot."""
        json_match = re.search(r'\{[^{}]*"name"\s*:', content)
        if json_match:
            start = json_match.start()
            brace_count = 0
            end = start
            for i in range(start, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            json_str = content[start:end]
            try:
                json.loads(json_str)
                return f"<json>{json_str}</json>"
            except json.JSONDecodeError:
                pass

        fallback = json.dumps(
            {"name": RESPOND_ACTION_NAME, "arguments": {"content": content.strip()}}
        )
        return f"<json>{fallback}</json>"

    def clear_session(self, context_id: str) -> None:
        """Remove session state for a completed task."""
        self._sessions.pop(context_id, None)


class _Session:
    """Per-context session state."""

    __slots__ = ("messages", "tools", "native_mode")

    def __init__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        native_mode: bool,
    ):
        self.messages = messages
        self.tools = tools
        self.native_mode = native_mode


# Fallback system prompt for when tool parsing fails (text-prompting mode)
_SYSTEM_PROMPT_FALLBACK = """\
You are a helpful agent. Follow the policy strictly and use the available tools \
to fulfill the user's request.

RESPONSE FORMAT (mandatory):
- Respond with exactly one JSON object wrapped in <json>...</json> tags.
- The JSON must have "name" (tool name or "respond") and "arguments" (dict).
- Use ONE tool at a time. Do NOT include any text outside the <json>...</json> tags.
- To reply to the user: <json>{"name": "respond", "arguments": {"content": "your message"}}</json>

RULES:
- Be PROACTIVE: look up information yourself rather than asking the user.
- Minimize back-and-forth. Gather data via tool calls BEFORE responding.
- Only ask the user for information you cannot find via tools.
- ALWAYS execute write operations after user confirms.
- Keep responses concise.
"""
