"""tau2-bench protocol adapter — multi-turn tool-calling via <json> tags.

tau2-bench's green agent sends plain text messages via A2A TextParts.
The first message contains:
- Policy text
- Tool list as JSON
- Instructions to respond with <json>{"name": "...", "arguments": {...}}</json>

Subsequent messages are tool call results or user observations as plain text.
The special action name "respond" sends a direct text reply to the user.

Reference: agentify_tau_bench/green_agent/agent.py in tau2-bench repo.
"""

from __future__ import annotations

import json
import logging
import os
import re

from openai import AsyncAzureOpenAI, AsyncOpenAI

from usage import tracker

logger = logging.getLogger("agentwhetters.tau2_adapter")

# Detection pattern: the green agent always includes this instruction
_TAU2_DETECTION_PATTERN = re.compile(
    r"Please wrap the JSON part with <json>", re.IGNORECASE
)

# Also check for tool list format
_TAU2_TOOL_LIST_PATTERN = re.compile(
    r"Here's a list of tools you can use", re.IGNORECASE
)

RESPOND_ACTION_NAME = "respond"


def is_tau2_protocol_message(text: str) -> bool:
    """Detect if a message is a tau2-bench first turn (contains tool instructions)."""
    return bool(
        _TAU2_DETECTION_PATTERN.search(text)
        and _TAU2_TOOL_LIST_PATTERN.search(text)
    )


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


_SYSTEM_PROMPT = """\
You are a customer service agent. Follow the policy strictly and use the \
available tools to fulfill the user's request.

RESPONSE FORMAT (mandatory):
- Respond with exactly one JSON object wrapped in <json>...</json> tags.
- The JSON must have "name" (tool name or "respond") and "arguments" (dict).
- Use ONE tool at a time. Do NOT include any text outside the <json>...</json> tags.
- To reply to the user: <json>{"name": "respond", "arguments": {"content": "your message"}}</json>

EFFICIENCY RULES (critical for success):
- Be PROACTIVE: look up information yourself rather than asking the user for it.
- When you have a user ID, immediately call get_user_details to find their info.
- When searching for a specific reservation, check ALL candidate reservations \
yourself by calling tools sequentially rather than asking the user which one to check.
- Minimize back-and-forth. Gather all needed data via tool calls BEFORE responding.
- Only ask the user for information you genuinely cannot find via tools.
- When the policy requires explicit user confirmation (e.g., before modifying a \
booking), ask ONCE with all relevant details, then proceed on confirmation.
- Keep responses concise. Do not repeat information the user already provided.
"""


class Tau2Adapter:
    """Handles tau2-bench multi-turn tool-calling protocol."""

    def __init__(self):
        self._client = _make_client()
        self._model = os.environ.get("AGENT_MODEL", "gpt-5.4")
        # context_id -> conversation history
        self._sessions: dict[str, list[dict[str, str]]] = {}

    async def handle_turn(self, context_id: str, text: str) -> str:
        """Process a tau2-bench turn and return the response text.

        Args:
            context_id: Session key for multi-turn state.
            text: The raw text message from the green agent.

        Returns:
            Response text containing <json>...</json> tags.
        """
        if context_id not in self._sessions:
            # First turn: includes policy + tools + initial user message
            self._sessions[context_id] = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]
        else:
            # Subsequent turns: tool results or observations
            self._sessions[context_id].append(
                {"role": "user", "content": text}
            )

        messages = self._sessions[context_id]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
            )
        except Exception as exc:
            logger.exception("tau2 adapter LLM call failed: %s", exc)
            # Return a safe respond action on error
            fallback = json.dumps(
                {"name": RESPOND_ACTION_NAME, "arguments": {"content": f"Error: {exc}"}}
            )
            return f"<json>{fallback}</json>"

        tracker.record(response, label="tau2_adapter")

        content = response.choices[0].message.content or ""

        # Store assistant response in history
        self._sessions[context_id].append(
            {"role": "assistant", "content": content}
        )

        # Ensure response has <json> tags with valid JSON inside
        if "<json>" not in content:
            content = self._fix_response_format(content)
        else:
            # Validate JSON inside existing tags
            content = self._validate_json_in_tags(content)

        return content

    def _validate_json_in_tags(self, content: str) -> str:
        """Validate JSON inside <json> tags and fix common LLM errors."""
        match = re.search(r"<json>(.*?)</json>", content, re.DOTALL)
        if not match:
            return content
        json_str = match.group(1).strip()
        try:
            json.loads(json_str)
            return content  # valid JSON, return as-is
        except json.JSONDecodeError:
            # Try to extract valid JSON via brace counting
            fixed = self._extract_valid_json(json_str)
            if fixed:
                return f"<json>{fixed}</json>"
            # Fall back to full _fix_response_format on the raw content
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
        # Try to find a JSON object in the response
        json_match = re.search(r'\{[^{}]*"name"\s*:', content)
        if json_match:
            # Find the full JSON object
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
                json.loads(json_str)  # validate
                return f"<json>{json_str}</json>"
            except json.JSONDecodeError:
                pass

        # Last resort: wrap a respond action with whatever content we have
        fallback = json.dumps(
            {"name": RESPOND_ACTION_NAME, "arguments": {"content": content.strip()}}
        )
        return f"<json>{fallback}</json>"

    def clear_session(self, context_id: str) -> None:
        """Remove session state for a completed task."""
        self._sessions.pop(context_id, None)
