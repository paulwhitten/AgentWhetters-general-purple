#!/usr/bin/env python3
"""Local test script for the AgentWhetters general agent.

Usage:
    # Start the server first:
    cd src && python server.py --port 9019

    # Then run this script:
    python tests/test_local.py

Requires OPENAI_API_KEY (or Azure config) to be set in .env or environment.
"""

import asyncio
import sys

import httpx

AGENT_URL = "http://localhost:9019"
TIMEOUT = 300  # 5 minutes per task


async def send_task(text: str, context_id: str | None = None) -> dict:
    """Send a task via A2A JSON-RPC and return the response."""
    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "id": f"test-{id(text)}",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": text}],
                "messageId": f"msg-{id(text)}",
            }
        },
    }
    if context_id:
        payload["params"]["message"]["contextId"] = context_id

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{AGENT_URL}/", json=payload)
        return resp.json()


async def run_tests():
    """Run a set of simple validation tasks."""
    print("=" * 60)
    print("AgentWhetters Local Test Suite")
    print("=" * 60)

    # Test 1: Agent card
    print("\n[1] Testing agent card...")
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{AGENT_URL}/.well-known/agent-card.json")
        card = resp.json()
        assert card["name"] == "AgentWhetters_general", f"Got: {card['name']}"
        print(f"    OK: Agent card served, protocol {card['protocolVersion']}")

    # Test 2: Simple arithmetic
    print("\n[2] Testing simple arithmetic (2+2)...")
    result = await send_task("What is 2+2? Reply with just the number, nothing else.")
    answer = extract_answer(result)
    print(f"    Answer: {answer}")
    assert "4" in answer, f"Expected '4', got: {answer}"
    print("    OK")

    # Test 3: Shell command
    print("\n[3] Testing shell access (ls /)...")
    result = await send_task("Run 'ls /' and return the output exactly as printed.")
    answer = extract_answer(result)
    print(f"    Answer (first 200 chars): {answer[:200]}")
    assert "bin" in answer or "usr" in answer, "Expected directory listing"
    print("    OK")

    # Test 4: Python computation
    print("\n[4] Testing Python computation...")
    result = await send_task(
        "Use python3 to compute the sum of all prime numbers below 100. "
        "Reply with just the number."
    )
    answer = extract_answer(result)
    print(f"    Answer: {answer}")
    assert "1060" in answer, f"Expected '1060', got: {answer}"
    print("    OK")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


def extract_answer(response: dict) -> str:
    """Extract the text answer from an A2A response."""
    result = response.get("result", {})
    # Check artifacts first
    artifacts = result.get("artifacts", [])
    for artifact in artifacts:
        for part in artifact.get("parts", []):
            if part.get("kind") == "text":
                return part["text"]
    # Fall back to history
    history = result.get("history", [])
    for msg in reversed(history):
        if msg.get("role") == "agent":
            for part in msg.get("parts", []):
                if part.get("kind") == "text":
                    return part["text"]
    return ""


if __name__ == "__main__":
    try:
        asyncio.run(run_tests())
    except httpx.ConnectError:
        print(f"ERROR: Cannot connect to {AGENT_URL}")
        print("Start the server first: cd src && python server.py --port 9019")
        sys.exit(1)
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
