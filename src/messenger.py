"""A2A client utilities for communicating with other agents."""

from __future__ import annotations

import json
import logging
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import DataPart, Message, Part, Role, TextPart

logger = logging.getLogger("agentwhetters.messenger")

DEFAULT_TIMEOUT = 300


def create_message(
    *, role: Role = Role.user, text: str, context_id: str | None = None
) -> Message:
    """Create an A2A message with text content."""
    return Message(
        kind="message",
        role=role,
        parts=[Part(root=TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def merge_parts(parts: list[Part]) -> str:
    """Extract text from message parts."""
    chunks = []
    for part in parts:
        root = part.root if hasattr(part, "root") else part
        if isinstance(root, TextPart):
            chunks.append(root.text)
        elif isinstance(root, DataPart):
            chunks.append(json.dumps(root.data, indent=2))
    return "\n".join(chunks)


async def send_message(
    message: str,
    base_url: str,
    context_id: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """Send a message to an agent and return the response."""
    timeout_config = httpx.Timeout(timeout=None, read=None, write=None, connect=60.0)
    async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=True)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        outbound_msg = create_message(text=message, context_id=context_id)
        last_event = None
        outputs: dict = {"response": "", "context_id": None}

        async for event in client.send_message(outbound_msg):
            last_event = event

        match last_event:
            case Message() as msg:
                outputs["context_id"] = msg.context_id
                outputs["response"] = merge_parts(msg.parts)

            case (task, _update):
                outputs["context_id"] = task.context_id
                if task.status and task.status.message:
                    outputs["response"] = merge_parts(task.status.message.parts)
                if task.artifacts:
                    artifact_parts = []
                    for artifact in task.artifacts:
                        artifact_parts.append(merge_parts(artifact.parts))
                    outputs["artifacts"] = artifact_parts
                    # Use last artifact as primary response if no status message
                    if not outputs["response"] and artifact_parts:
                        outputs["response"] = artifact_parts[-1]

        return outputs
