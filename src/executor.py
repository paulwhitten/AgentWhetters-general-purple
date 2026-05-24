"""Custom A2A AgentExecutor that routes messages to protocol adapters.

Replaces ADK's to_a2a() abstraction to give us full control over
TaskStatusUpdateEvent and TaskArtifactUpdateEvent. Routes incoming
messages to the appropriate protocol adapter based on message format
(not benchmark identity).
"""

from __future__ import annotations

import base64
import logging
import uuid
from datetime import datetime, timezone

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    Artifact,
    DataPart,
    FilePart,
    FileWithBytes,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from agent import Agent
from shell_adapter import ShellProtocolAdapter, is_shell_protocol_message
from toolchat_adapter import ToolChatAdapter

logger = logging.getLogger("agentwhetters.executor")


def _extract_text(message: Message) -> str:
    """Extract all text content from an A2A message."""
    parts = []
    for part in message.parts:
        if isinstance(part.root, TextPart):
            parts.append(part.root.text)
    return "\n".join(parts)


import re

# Patterns indicating a self-contained prompt that should bypass our agentic wrapper.
# These prompts already include full instructions and expect a direct LLM response.
_PASSTHROUGH_PATTERNS = [
    re.compile(r"Output Format:", re.IGNORECASE),
    re.compile(r"Response Format:", re.IGNORECASE),
    re.compile(r"Your task is to generate", re.IGNORECASE),
    re.compile(r"def process_graph|def solve|def answer", re.IGNORECASE),
]

_MIN_PASSTHROUGH_LENGTH = 2000


def _is_passthrough_prompt(text: str) -> bool:
    """Detect self-contained prompts that don't need shell tools or developer message.

    Heuristic: long prompts (>2000 chars) that contain explicit output format
    instructions or code generation patterns. These are typically benchmark
    prompts that should be forwarded directly to the LLM.
    """
    if len(text) < _MIN_PASSTHROUGH_LENGTH:
        return False
    matches = sum(1 for p in _PASSTHROUGH_PATTERNS if p.search(text))
    return matches >= 2


def _has_file_parts(message: Message) -> bool:
    """Check if message contains FileParts (indicator of CyberGym)."""
    for part in message.parts:
        if isinstance(part.root, FilePart):
            return True
    return False


def _has_data_part_with_exit_code(message: Message) -> bool:
    """Check if message is a CyberGym test result (DataPart with exit_code)."""
    for part in message.parts:
        if isinstance(part.root, DataPart):
            data = part.root.data
            if isinstance(data, dict) and ("exit_code" in data or "error" in data):
                return True
    return False


def _get_toolchat_data(message: Message) -> dict | None:
    """Extract tool-chat protocol data from a DataPart if present.

    Returns the data dict if the message contains a DataPart with
    'messages' or 'bootstrap' keys (tool-calling chat protocol).
    """
    for part in message.parts:
        if isinstance(part.root, DataPart):
            data = part.root.data
            if isinstance(data, dict) and ("messages" in data or "bootstrap" in data):
                return data
    return None


def _is_vuln_analysis_message(message: Message) -> bool:
    """Detect vulnerability-analysis protocol messages (FileParts + exit_code DataParts)."""
    return _has_file_parts(message) or _has_data_part_with_exit_code(message)


class PurpleAgentExecutor(AgentExecutor):
    """Routes A2A messages to protocol adapters based on message format."""

    def __init__(self):
        super().__init__()
        self._shell_adapter = ShellProtocolAdapter()
        self._toolchat_adapter = ToolChatAdapter()
        # context_id -> handler for multi-turn vuln-analysis sessions
        self._vuln_sessions: dict[str, object] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Dispatch to appropriate handler based on message content."""
        message = context.message
        logger.info("Execute called: task_id=%s context_id=%s", context.task_id, context.context_id)

        if message is None:
            await self._send_error(context, event_queue, "No message received")
            return

        logger.info("Message has %d parts", len(message.parts))
        for i, part in enumerate(message.parts):
            logger.info("  Part %d: %s", i, type(part.root).__name__)

        task_id = context.task_id
        context_id = context.context_id

        # Send initial "working" status
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.working,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                final=False,
            )
        )

        try:
            # Route to appropriate handler
            text = _extract_text(message)
            toolchat_data = _get_toolchat_data(message)

            if _is_vuln_analysis_message(message) or context_id in self._vuln_sessions:
                await self._handle_vuln_analysis(context, event_queue, message)
            elif toolchat_data is not None:
                await self._handle_toolchat(context, event_queue, toolchat_data)
            elif is_shell_protocol_message(text) or context_id in self._shell_adapter._sessions:
                await self._handle_shell(context, event_queue, text)
            else:
                await self._handle_standard(context, event_queue, text)

        except Exception as e:
            logger.exception("Error in executor: %s", e)
            await self._send_error(context, event_queue, str(e))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel an ongoing task."""
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.canceled,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                final=True,
            )
        )

    async def _handle_shell(
        self, context: RequestContext, event_queue: EventQueue, text: str,
    ) -> None:
        """Handle shell-command protocol messages."""
        session_key = context.context_id or "default"
        self._shell_adapter.get_session_messages(session_key, text)
        result = await self._shell_adapter.handle_turn(session_key, text)

        # Send result as artifact + completion
        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                last_chunk=True,
                append=False,
                artifact=Artifact(
                    artifact_id=str(uuid.uuid4()),
                    parts=[Part(root=TextPart(text=result))],
                ),
            )
        )
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.completed,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                final=True,
            )
        )

    async def _handle_standard(
        self, context: RequestContext, event_queue: EventQueue, text: str,
    ) -> None:
        """Handle standard single-shot tasks."""
        if not text:
            text = "(No input provided)"

        agent = Agent()

        # Use passthrough for self-contained prompts (code generation, etc.)
        if _is_passthrough_prompt(text):
            logger.info("Using passthrough mode (self-contained prompt detected)")
            result = await agent.passthrough(text)
        else:
            result = await agent.execute(text)

        # Send result as artifact + completion
        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                last_chunk=True,
                append=False,
                artifact=Artifact(
                    artifact_id=str(uuid.uuid4()),
                    parts=[Part(root=TextPart(text=result))],
                ),
            )
        )
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.completed,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                final=True,
            )
        )

    async def _handle_vuln_analysis(
        self, context: RequestContext, event_queue: EventQueue, message: Message,
    ) -> None:
        """Handle vulnerability-analysis protocol messages."""
        # Lazy import to avoid circular deps
        from vuln_adapter import VulnAnalysisAdapter

        context_id = context.context_id or "default"

        if context_id not in self._vuln_sessions:
            handler = VulnAnalysisAdapter()
            self._vuln_sessions[context_id] = handler
        else:
            handler = self._vuln_sessions[context_id]

        await handler.handle_message(context, event_queue, message)

    async def _handle_toolchat(
        self, context: RequestContext, event_queue: EventQueue, data: dict,
    ) -> None:
        """Handle tool-calling chat protocol requests.

        Supports bootstrap (cache context once) and regular turns (forward to LLM).
        Returns DataPart with either text content or tool_calls.
        """
        if data.get("bootstrap"):
            logger.info("Tool-chat adapter: bootstrap request")
            result_data = await self._toolchat_adapter.handle_bootstrap(data)
        else:
            logger.info("Tool-chat adapter: conversation turn")
            result_data = await self._toolchat_adapter.handle_turn(data)

        # Return result as DataPart in the status message
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.completed,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    message=Message(
                        message_id=str(uuid.uuid4()),
                        role=Role.agent,
                        parts=[Part(root=DataPart(data=result_data))],
                    ),
                ),
                final=True,
            )
        )

    async def _send_error(
        self, context: RequestContext, event_queue: EventQueue, error_msg: str,
    ) -> None:
        """Send an error status update."""
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.failed,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    message=Message(
                        message_id=str(uuid.uuid4()),
                        role=Role.agent,
                        parts=[Part(root=TextPart(text=f"Error: {error_msg}"))],
                    ),
                ),
                final=True,
            )
        )
