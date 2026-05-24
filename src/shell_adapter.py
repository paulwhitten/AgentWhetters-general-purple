"""Shell protocol adapter — interactive command execution via A2A.

Implements multi-turn A2A conversation for shell-command protocols.
Instead of executing commands locally, the agent returns exec_request
JSON and processes exec_result messages from the green agent which
runs commands in a container environment.

Uses the OpenAI Responses API items accumulation pattern: response.output
items are stored between turns, and exec_results are paired with their
function_call via call_id. This enables proper context caching and
compaction by the API.

Protocol:
  Green -> Purple: {"kind": "task", "protocol": "...-shell-v1", "instruction": "..."}
  Purple -> Green: {"kind": "exec_request", "command": "...", "timeout": N}
  Green -> Purple: {"kind": "exec_result", "exit_code": N, "stdout": "...", "stderr": "..."}
  ...
  Purple -> Green: {"kind": "final"}
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

from openai import AsyncAzureOpenAI, AsyncOpenAI

from tools import (
    DEFAULT_COMPACT_THRESHOLD,
    DEFAULT_STEP_LIMIT,
    DONE_TOOL,
    is_reasoning_model,
    truncate,
)
from usage import tracker

logger = logging.getLogger("agentwhetters.shell_adapter")

# Truncate long command output to keep context manageable
MAX_OUTPUT_CHARS = 30_000

# Tool definition for requesting remote execution
REMOTE_EXEC_TOOL: dict = {
    "type": "function",
    "name": "exec_command",
    "description": (
        "Execute a shell command in the remote task environment (Docker container). "
        "Returns stdout, stderr, and exit code. Chain commands with && or ;. "
        "Set timeout appropriately: 30s for quick commands, 120-300s for builds."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute in the remote environment.",
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait (1-300). Default 30.",
            },
        },
        "required": ["command", "timeout"],
        "additionalProperties": False,
    },
    "strict": True,
}


TB_DEVELOPER_MESSAGE = """\
You are an expert systems administrator and software engineer solving tasks \
in a Linux Docker container. You interact by requesting shell commands one at \
a time and receiving their output.

## Approach

1. Read the task instruction carefully before acting.
2. Explore the environment first: ls /, ls /app, cat README*, which <tool>, etc.
3. Plan your approach, then execute step by step.
4. Verify your work after each significant step.
5. When done and verified, call done.

## Efficiency

- Chain related commands: cmd1 && cmd2 && cmd3
- Write multi-step logic as inline scripts: bash -c '...' or python3 -c '...'
- Install packages in one shot: apt-get update && apt-get install -y pkg1 pkg2
- Pipe long output through head/tail/grep to keep it manageable.
- Set timeout appropriately: 30s for quick commands, 120-300s for builds/downloads.
- You have a limited turn budget. Be efficient and direct.

## Common patterns

- **Builds**: read Makefile/CMakeLists.txt, install dependencies, then build.
- **Git**: use git log --oneline, git reflog, git status to understand state.
- **Services**: check config syntax (nginx -t), then start, then verify (curl localhost).
- **Code fixes**: read the code, understand the bug, make minimal targeted changes, test.
- **File recovery**: check file headers, use appropriate tools (sqlite3 .recover, etc).
- **Data/ML**: check Python version, install deps with pip, run scripts.

## Rules

- Never guess at file contents -- always cat/read them.
- Read error messages carefully before retrying.
- If a command fails, diagnose why before trying alternatives.
- If an approach fails 2-3 times, try a fundamentally different strategy.
- When the task says to produce a specific file or output, verify it exists and is correct.
- When finished, call `done` with a brief summary of what you accomplished.
"""


def is_shell_protocol_message(text: str) -> bool:
    """Check if an incoming message uses a shell-command protocol."""
    try:
        payload = json.loads(text)
        return (
            isinstance(payload, dict)
            and payload.get("protocol") == "terminal-bench-shell-v1"
        )
    except (json.JSONDecodeError, TypeError):
        return False


def parse_shell_message(text: str) -> dict:
    """Parse a shell protocol message."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"kind": "unknown", "raw": text}


def _make_client() -> AsyncOpenAI:
    """Create OpenAI client (same logic as agent.py)."""
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


@dataclass
class _ShellSessionState:
    """Persistent state for a shell-command protocol session.

    Uses proper Responses API items accumulation: stores the full items list
    (including response.output) so the API can leverage prefix caching and
    compaction across turns.
    """

    instruction: str = ""
    # Full OpenAI Responses API items list (input + response outputs)
    items: list = field(default_factory=list)
    # call_id of the pending exec_command function_call awaiting output
    pending_call_id: str | None = None
    turn_count: int = 0


class ShellProtocolAdapter:
    """Handles multi-turn shell-command protocol sessions.

    Maintains conversation state in a class-level dictionary keyed by
    session context, since ADK session events may not accumulate reliably
    across A2A messages.
    """

    # Class-level state: maps session_key -> _ShellSessionState
    _sessions: dict[str, _ShellSessionState] = {}

    def __init__(self):
        self.client = _make_client()
        self.model = os.environ.get("AGENT_MODEL", "gpt-5.4")
        self.step_limit = int(os.environ.get("AGENT_STEP_LIMIT", str(DEFAULT_STEP_LIMIT)))
        self.compact_threshold = int(
            os.environ.get("AGENT_COMPACT_THRESHOLD", str(DEFAULT_COMPACT_THRESHOLD))
        )

    def get_session_messages(self, session_key: str, new_message: str) -> list[str]:
        """Accumulate raw messages for a session (for backward compat with server.py)."""
        # We still track raw messages for logging/debugging, but the real state
        # is in _sessions[session_key].items
        if not hasattr(self, "_raw_messages"):
            self._raw_messages: dict[str, list[str]] = {}
        if session_key not in self._raw_messages:
            self._raw_messages[session_key] = []
        self._raw_messages[session_key].append(new_message)
        return list(self._raw_messages[session_key])

    async def handle_turn(self, session_key: str, message_text: str) -> str:
        """Process one turn of the TB protocol using items accumulation.

        Args:
            session_key: Unique session identifier.
            message_text: Raw JSON message from green agent.

        Returns:
            JSON string: either exec_request or final.
        """
        payload = parse_shell_message(message_text)

        # Get or create session state
        if session_key not in self._sessions:
            # First turn: task instruction
            state = _ShellSessionState()
            state.instruction = payload.get("instruction", message_text)
            state.items = [
                {"role": "user", "content": state.instruction},
            ]
            self._sessions[session_key] = state
            logger.info("TB new session: %d chars instruction", len(state.instruction))
        else:
            # Continuation: exec_result
            state = self._sessions[session_key]
            state.turn_count += 1

            exit_code = payload.get("exit_code", -1)
            stdout = payload.get("stdout", "")
            stderr = payload.get("stderr", "")

            # Truncate large outputs to keep context manageable
            stdout = truncate(stdout, MAX_OUTPUT_CHARS)
            stderr = truncate(stderr, MAX_OUTPUT_CHARS)

            # Build tool output text
            result_text = f"exit_code={exit_code}\n"
            if stdout:
                result_text += f"stdout:\n{stdout}\n"
            if stderr:
                result_text += f"stderr:\n{stderr}\n"
            if not stdout and not stderr:
                result_text += "(no output)\n"

            # Pair with the pending function_call via call_id
            if state.pending_call_id:
                state.items.append({
                    "type": "function_call_output",
                    "call_id": state.pending_call_id,
                    "output": result_text,
                })
                state.pending_call_id = None
            else:
                # Fallback: append as user message
                state.items.append({
                    "role": "user",
                    "content": f"Command result:\n{result_text}",
                })

        logger.info("TB turn %d, %d items in context",
                    state.turn_count + 1, len(state.items))

        # Check step budget
        tb_step_limit = min(self.step_limit, 30)
        remaining = tb_step_limit - state.turn_count
        if remaining <= 0:
            logger.warning("TB step budget exhausted")
            tracker.log_summary()
            return json.dumps({"kind": "final"})

        # Inject step warning via developer message appended to items
        extra_items: list = []
        if remaining == 5:
            extra_items.append({
                "role": "developer",
                "content": (
                    "[SYSTEM: You have 5 exec calls remaining. Wrap up your work now. "
                    "If the task is solved or mostly solved, call `done` immediately.]"
                ),
            })
        elif remaining <= 2:
            extra_items.append({
                "role": "developer",
                "content": (
                    "[SYSTEM: FINAL STEP. Call `done` NOW with what you've accomplished. "
                    "Do not run more commands.]"
                ),
            })

        reasoning = is_reasoning_model(self.model)
        tools = [REMOTE_EXEC_TOOL, DONE_TOOL]

        api_kwargs: dict = {
            "model": self.model,
            "instructions": TB_DEVELOPER_MESSAGE,
            "input": state.items + extra_items,
            "tools": tools,
            "parallel_tool_calls": False,
            "store": False,
        }
        if reasoning:
            api_kwargs["include"] = ["reasoning.encrypted_content"]
            api_kwargs["reasoning"] = {"effort": "high", "summary": "auto"}
            api_kwargs["max_output_tokens"] = 16_000
        else:
            api_kwargs["temperature"] = 0.0
            api_kwargs["max_output_tokens"] = 4096

        try:
            response = await self.client.responses.create(**api_kwargs)
        except Exception as exc:
            logger.error("TB API error: %s", exc)
            return json.dumps({"kind": "final"})

        tracker.record(response, label=f"tb-turn={state.turn_count+1}")

        # Accumulate response output items into state for next turn
        state.items.extend(response.output)

        # Extract function calls from response
        function_calls = [it for it in response.output if it.type == "function_call"]

        if not function_calls:
            # Model returned text without tool call - retry once with a nudge
            logger.info("TB model returned text without tool call, retrying with nudge")
            state.items.append({
                "role": "developer",
                "content": (
                    "You must call either `exec_command` to run a shell command or "
                    "`done` to finish the task. Do not respond with plain text. "
                    "If you are finished, call `done` now with a summary."
                ),
            })
            try:
                api_kwargs["input"] = state.items
                response = await self.client.responses.create(**api_kwargs)
            except Exception as exc:
                logger.error("TB API retry error: %s", exc)
                return json.dumps({"kind": "final"})

            tracker.record(response, label=f"tb-turn={state.turn_count+1}-retry")
            state.items.extend(response.output)
            function_calls = [it for it in response.output if it.type == "function_call"]

        if not function_calls:
            # Still no tool call after retry - give up
            logger.info("TB model still no tool call after retry, treating as final")
            tracker.log_summary()
            del self._sessions[session_key]
            return json.dumps({"kind": "final"})

        for fc in function_calls:
            try:
                args = json.loads(fc.arguments)
            except json.JSONDecodeError:
                args = {}

            if fc.name == "exec_command":
                command = args.get("command", "echo 'no command'")
                timeout = min(max(args.get("timeout", 30), 1), 300)
                state.pending_call_id = fc.call_id
                logger.info("TB exec_request: %s (timeout=%d)", command[:120], timeout)
                return json.dumps({
                    "kind": "exec_request",
                    "command": command,
                    "timeout": timeout,
                })
            elif fc.name == "done":
                answer = args.get("answer", "Task completed.")
                logger.info("TB done: %s", answer[:100])
                tracker.log_summary()
                # Clean up session
                del self._sessions[session_key]
                return json.dumps({"kind": "final"})

        # Shouldn't reach here, but safety fallback
        logger.warning("TB unexpected state: function_calls present but unhandled")
        tracker.log_summary()
        del self._sessions[session_key]
        return json.dumps({"kind": "final"})
