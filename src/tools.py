"""Shared tool definitions, shell execution, and utilities."""

from __future__ import annotations

import asyncio
import os
import signal

# ---------------------------------------------------------------------------
# Constants (configurable via env vars)
# ---------------------------------------------------------------------------

DEFAULT_STEP_LIMIT = 50
DEFAULT_COMMAND_TIMEOUT = 60
DEFAULT_TOOL_RESULT_LIMIT = 30_000
DEFAULT_COMPACT_THRESHOLD = 200_000

_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def is_reasoning_model(model_name: str) -> bool:
    """Check if a model name corresponds to a reasoning-class model."""
    return any(model_name.startswith(p) for p in _REASONING_MODEL_PREFIXES)


# ---------------------------------------------------------------------------
# Tool definitions for the OpenAI Responses API
# ---------------------------------------------------------------------------

SHELL_TOOL: dict = {
    "type": "shell",
    "environment": {"type": "local"},
}

RUN_COMMAND_TOOL: dict = {
    "type": "function",
    "name": "run_command",
    "description": "Execute a shell command. Returns stdout, stderr, and exit code.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute (bash).",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    "strict": True,
}

DONE_TOOL: dict = {
    "type": "function",
    "name": "done",
    "description": "Signal that the task is complete. Provide the final answer.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The final answer or result to return.",
            },
        },
        "required": ["answer"],
        "additionalProperties": False,
    },
    "strict": True,
}

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def truncate(text: str, limit: int) -> str:
    """Truncate text preserving head and tail."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return (
        text[:head]
        + f"\n... (truncated, {len(text)} total chars) ...\n"
        + text[-tail:]
    )


# ---------------------------------------------------------------------------
# Shell execution
# ---------------------------------------------------------------------------


async def run_shell_command(
    command: str,
    timeout: int | None = None,
) -> dict:
    """Execute a shell command via subprocess and return structured result."""
    if timeout is None:
        timeout = int(os.environ.get("AGENT_COMMAND_TIMEOUT", str(DEFAULT_COMMAND_TIMEOUT)))

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # process group for clean kill
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            # Kill entire process group to handle child processes
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                proc.kill()
            await proc.communicate()
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "exit_code": -1,
            }
        return {
            "stdout": (stdout_bytes or b"").decode("utf-8", errors="replace"),
            "stderr": (stderr_bytes or b"").decode("utf-8", errors="replace"),
            "exit_code": proc.returncode or 0,
        }
    except Exception as exc:
        return {"stdout": "", "stderr": f"Error: {exc}", "exit_code": -1}
