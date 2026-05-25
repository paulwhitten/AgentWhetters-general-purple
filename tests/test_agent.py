"""Basic tests for the A2A agent."""

import sys
from pathlib import Path

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_agent_imports():
    """Test that all agent modules import correctly."""
    from agent import Agent, DEVELOPER_MESSAGE, make_openai_client  # noqa: F401
    from tools import SHELL_TOOL, DONE_TOOL, RUN_COMMAND_TOOL, is_reasoning_model  # noqa: F401
    from server import create_app  # noqa: F401

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
        assert agent.step_limit == 50
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


def test_tau2_protocol_detection():
    """Test that tau2-bench messages are correctly detected."""
    from tau2_adapter import is_tau2_protocol_message

    # Simulated tau2 first message (contains both detection patterns)
    tau2_msg = (
        "Some policy text...\n"
        "Here's a list of tools you can use (you can use at most one tool at a time):\n"
        '[{"type": "function", "function": {"name": "find_user"}}]\n'
        "Please response in the JSON format. "
        "Please wrap the JSON part with <json>...</json> tags.\n"
    )
    assert is_tau2_protocol_message(tau2_msg) is True

    # Normal message should not match
    assert is_tau2_protocol_message("Hello, can you help me?") is False
    assert is_tau2_protocol_message("run ls -la") is False

    # Partial match (only one pattern) should not trigger
    assert is_tau2_protocol_message(
        "Please wrap the JSON part with <json>...</json> tags."
    ) is False


def test_tau2_response_fix():
    """Test that the adapter fixes malformed responses."""
    from tau2_adapter import Tau2Adapter

    adapter = Tau2Adapter()

    # Already correct
    correct = '<json>{"name": "respond", "arguments": {"content": "Hi"}}</json>'
    assert "<json>" in correct  # no fix needed

    # Missing tags but valid JSON
    raw_json = '{"name": "find_user", "arguments": {"id": "123"}}'
    fixed = adapter._fix_response_format(raw_json)
    assert "<json>" in fixed
    assert "</json>" in fixed
    assert '"find_user"' in fixed


def test_tau2_validate_json_in_tags():
    """Test that _validate_json_in_tags fixes malformed JSON inside tags."""
    from tau2_adapter import Tau2Adapter

    adapter = Tau2Adapter()

    # Valid JSON in tags - should pass through unchanged
    valid = '<json>{"name": "respond", "arguments": {"content": "Hi"}}</json>'
    assert adapter._validate_json_in_tags(valid) == valid

    # Extra closing brace (common LLM error)
    extra_brace = '<json>{"name":"get_user_details","arguments":{"user_id":"raj_sanchez_7340"}}}</json>'
    fixed = adapter._validate_json_in_tags(extra_brace)
    assert "<json>" in fixed
    assert "</json>" in fixed
    # Should extract valid JSON via brace counting
    import json
    inner = fixed.replace("<json>", "").replace("</json>", "")
    parsed = json.loads(inner)
    assert parsed["name"] == "get_user_details"
    assert parsed["arguments"]["user_id"] == "raj_sanchez_7340"


def test_tau2_extract_valid_json():
    """Test brace-counting JSON extraction."""
    from tau2_adapter import Tau2Adapter

    adapter = Tau2Adapter()

    # Normal JSON
    assert adapter._extract_valid_json('{"a": 1}') == '{"a": 1}'

    # Extra braces after
    assert adapter._extract_valid_json('{"a": {"b": 2}}}') == '{"a": {"b": 2}}'

    # No JSON
    assert adapter._extract_valid_json('no json here') is None

    # Nested JSON
    result = adapter._extract_valid_json('{"name": "tool", "arguments": {"x": 1}}')
    assert result is not None
    import json
    assert json.loads(result)["name"] == "tool"
