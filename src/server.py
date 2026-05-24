"""A2A server entry point using the raw a2a SDK.

Uses a custom AgentExecutor that routes messages to Terminal Bench,
CyberGym, structured chat (Pi-Bench), or standard handlers.
This gives us full control over TaskStatusUpdateEvent and
TaskArtifactUpdateEvent (required for CyberGym's multi-turn protocol
with DataPart + FilePart).
"""

import argparse
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotificationConfigStore, InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from executor import PurpleAgentExecutor

load_dotenv()

_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"

logging.basicConfig(
    level=os.environ.get("AGENT_LOG_LEVEL", "INFO"),
    format=_LOG_FORMAT,
)

# File logging
_log_dir = Path("logs")
_log_dir.mkdir(exist_ok=True)
_log_file = _log_dir / f"agent-{datetime.now():%Y%m%d-%H%M%S}.log"
_file_handler = logging.FileHandler(_log_file)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
_file_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger("agentwhetters")
logger.info("Logging to %s", _log_file)


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------


def _build_agent_card(host: str, port: int) -> AgentCard:
    """Build the agent card for service discovery."""
    return AgentCard(
        name="AgentWhetters_general_purple",
        description=(
            "General-purpose purple agent for AgentX-AgentBeats Sprint 4. "
            "Adapts across coding, research, cybersecurity, finance, "
            "safety, and other task categories without benchmark-specific logic."
        ),
        url=f"http://{host}:{port}/",
        version="1.0.1",
        capabilities=AgentCapabilities(streaming=True),
        defaultInputModes=["text/plain", "application/octet-stream"],
        defaultOutputModes=["text/plain", "application/octet-stream"],
        skills=[
            AgentSkill(
                id="general",
                name="General Problem Solving",
                description="Solves coding, research, cybersecurity, and other tasks",
                tags=["general", "coding", "cybersecurity"],
            ),
        ],
    )


# Bootstrap extension URI for policy-compliance benchmarks
POLICY_BOOTSTRAP_EXTENSION = "urn:pi-bench:policy-bootstrap:v1"


def _build_agent_card_json(host: str, port: int) -> dict:
    """Build agent card JSON with extensions (not supported by SDK AgentCard type)."""
    card = _build_agent_card(host, port)
    card_dict = card.model_dump(by_alias=True, exclude_none=True)
    # Add extensions for benchmark protocol support
    card_dict["extensions"] = [POLICY_BOOTSTRAP_EXTENSION]
    return card_dict


def _make_agent_card_route(host: str, port: int) -> Route:
    """Create a custom route that returns the agent card with extensions."""
    card_json = _build_agent_card_json(host, port)

    async def agent_card_handler(request: Request) -> JSONResponse:
        return JSONResponse(card_json)

    return Route("/.well-known/agent.json", endpoint=agent_card_handler)


def _make_health_route() -> Route:
    """Create a health check endpoint."""
    async def health_handler(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return Route("/health", endpoint=health_handler)


class MessageIdMiddleware:
    """ASGI middleware that injects messageId into A2A message/send requests.

    Some benchmark harnesses (e.g. Pi-Bench) omit the messageId field which
    the a2a-sdk requires. This middleware patches it in transparently.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        # Buffer the request body
        body_parts: list[bytes] = []
        request_complete = False

        async def buffered_receive():
            nonlocal request_complete
            if request_complete:
                # Return empty body for subsequent reads
                return {"type": "http.request", "body": b"", "more_body": False}
            msg = await receive()
            body_parts.append(msg.get("body", b""))
            if not msg.get("more_body", False):
                request_complete = True
            return msg

        # Read the full body first
        while not request_complete:
            await buffered_receive()

        body = b"".join(body_parts)
        modified = False

        try:
            data = json.loads(body)
            if (
                isinstance(data, dict)
                and data.get("method") == "message/send"
                and "params" in data
            ):
                msg = data["params"].get("message", {})
                if isinstance(msg, dict) and "messageId" not in msg:
                    msg["messageId"] = str(uuid.uuid4())
                    modified = True
                # Also ensure configuration.taskId exists
                config = data["params"].setdefault("configuration", {})
                if isinstance(config, dict) and "taskId" not in config:
                    config["taskId"] = str(uuid.uuid4())
                    modified = True
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        if modified:
            body = json.dumps(data).encode()

        # Replay the body
        body_sent = False

        async def replay_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay_receive, send)


def create_app(host: str = "0.0.0.0", port: int = 9009) -> Starlette:
    """Create the A2A Starlette application with our custom executor."""
    agent_card = _build_agent_card(host, port)
    executor = PurpleAgentExecutor()

    task_store = InMemoryTaskStore()
    push_config_store = InMemoryPushNotificationConfigStore()

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
        push_config_store=push_config_store,
    )

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # Custom agent card route with extensions (must be first to take precedence)
    custom_routes = [_make_agent_card_route(host, port), _make_health_route()]

    @asynccontextmanager
    async def lifespan(app: Starlette):
        a2a_app.add_routes_to_app(app)
        yield

    app = Starlette(
        routes=custom_routes,
        lifespan=lifespan,
        middleware=[Middleware(MessageIdMiddleware)],
    )
    return app


def main():
    parser = argparse.ArgumentParser(description="AgentWhetters General Purple Agent")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind")
    args = parser.parse_args()

    logger.info("Starting AgentWhetters on %s:%d", args.host, args.port)

    app = create_app(host=args.host, port=args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
