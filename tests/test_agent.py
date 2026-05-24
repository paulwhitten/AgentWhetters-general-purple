"""Basic tests for the A2A agent."""

import sys
from pathlib import Path

import pytest

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_agent_imports():
    """Test that all agent modules import correctly."""
    from agent import Agent, DEVELOPER_MESSAGE, make_openai_client
    from tools import SHELL_TOOL, DONE_TOOL, RUN_COMMAND_TOOL, is_reasoning_model
    from server import GeneralPurposeAgent, create_app

    assert Agent is not None
    assert len(DEVELOPER_MESSAGE) > 500


def test_developer_message_structure():
    """Test that the developer message follows outcome-first structure."""
    from agent import DEVELOPER_MESSAGE

    # Should have key sections per OpenAI GPT-5.x guidance
    assert "# Goal" in DEVELOPER_MESSAGE
    assert "# Success criteria" in DEVELOPER_MESSAGE
    assert "# Constraints" in DEVELOPER_MESSAGE
    assert "# Output" in DEVELOPER_MESSAGE
    assert "# Stop rules" in DEVELOPER_MESSAGE
    # Should NOT have step-by-step instructions (anti-pattern for reasoning models)
    assert "Step 1" not in DEVELOPER_MESSAGE
    assert "Step 2" not in DEVELOPER_MESSAGE


def test_reasoning_model_detection():
    """Test that reasoning model detection works for current models."""
    from tools import is_reasoning_model

    assert is_reasoning_model("gpt-5.4") is True
    assert is_reasoning_model("gpt-5.4-mini") is True
    assert is_reasoning_model("o1-preview") is True
    assert is_reasoning_model("o3-mini") is True
    assert is_reasoning_model("o4-mini") is True
    assert is_reasoning_model("gpt-4o") is False
    assert is_reasoning_model("gpt-4.1") is False


def test_tool_definitions():
    """Test that tool definitions are well-formed."""
    from tools import SHELL_TOOL, DONE_TOOL, RUN_COMMAND_TOOL

    assert SHELL_TOOL["type"] == "shell"
    assert DONE_TOOL["type"] == "function"
    assert DONE_TOOL["name"] == "done"
    assert DONE_TOOL["strict"] is True
    assert RUN_COMMAND_TOOL["type"] == "function"
    assert RUN_COMMAND_TOOL["name"] == "run_command"
    assert RUN_COMMAND_TOOL["strict"] is True


def test_agent_defaults():
    """Test that Agent has sensible defaults."""
    import os
    # Clear env vars to test defaults
    env_backup = {k: os.environ.pop(k, None) for k in [
        "AGENT_MODEL", "AGENT_STEP_LIMIT", "AZURE_OPENAI_ENDPOINT"
    ]}
    try:
        from agent import Agent
        agent = Agent()
        assert agent.model == "gpt-5.4"
        assert agent.step_limit == 30
        assert agent.tool_result_limit == 30_000
        assert agent.compact_threshold == 200_000
    finally:
        for k, v in env_backup.items():
            if v is not None:
                os.environ[k] = v


def test_create_app():
    """Test that ADK app creation works."""
    from server import create_app
    app = create_app(host="localhost", port=9999)
    assert app is not None
