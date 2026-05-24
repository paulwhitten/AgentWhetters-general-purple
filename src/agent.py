"""General-purpose agent: single agentic loop using OpenAI Responses API.

No task classification or skill dispatching. The model has shell access
and decides how to solve any task type autonomously.

Prompt design follows OpenAI's GPT-5.x best practices:
- Outcome-first (define WHAT, not HOW)
- Developer role message for caching benefits
- Reasoning effort tuning
- Context compaction for long tasks
"""

from __future__ import annotations

import json
import logging
import os

from openai import AsyncAzureOpenAI, AsyncOpenAI

from tools import (
    DEFAULT_COMPACT_THRESHOLD,
    DEFAULT_STEP_LIMIT,
    DEFAULT_TOOL_RESULT_LIMIT,
    DONE_TOOL,
    RUN_COMMAND_TOOL,
    SHELL_TOOL,
    is_reasoning_model,
    run_shell_command,
    truncate,
)
from usage import tracker

logger = logging.getLogger("agentwhetters.agent")

# ---------------------------------------------------------------------------
# Developer message: outcome-first prompt per OpenAI GPT-5.x guidance.
# Placed as developer role in input array for prompt caching benefits.
# Static content first, variable content (user task) last.
# ---------------------------------------------------------------------------

DEVELOPER_MESSAGE = """\
Role: General-purpose autonomous agent with full Linux shell access. \
You solve any task — coding, research, cybersecurity, finance, analysis, \
customer service, safety evaluation — by reasoning and executing commands.

# Goal
Resolve the user's task end to end and return the final answer via the `done` tool.

# Success criteria
- The output matches the exact format the task requests (tags, JSON, diff, plain text, etc.)
- Any code changes pass available tests or validations
- Numerical answers are computed, never guessed
- The final answer is complete, precise, and directly addresses the question

# Constraints
- You have a limited step budget — be efficient and strategic
- Use python3 for non-trivial computation or data processing
- If the task provides files or a working directory, explore them before acting
- Match output format exactly: pay attention to delimiters, units, structure, field names
- Do not fabricate data or URLs — verify through shell commands
- For multi-turn tasks, follow the protocol described in the task instructions

# Answer format
When the task is a factual question or asks for a specific value/answer, wrap your \
final answer in <FINAL_ANSWER> tags inside the `done` tool output:
  <FINAL_ANSWER>your answer here</FINAL_ANSWER>
This applies to research questions, numerical lookups, data analysis, and any task \
requesting a specific answer. For coding tasks (patches, diffs, code fixes), return \
the code output directly without FINAL_ANSWER tags unless instructed otherwise.

# Output
Return your final answer by calling the `done` tool. The answer field should contain \
only the requested output — no preamble, no explanation unless explicitly asked for.

# Stop rules
- If you solve the task, call `done` immediately — do not run extra verification steps \
unless the task requires them
- If an approach fails after 2-3 attempts, try a fundamentally different strategy
- If you are running low on steps, immediately call `done` with your best answer so far \
rather than continuing to search — a partial answer is better than no answer
- Never loop on the same failing command — diagnose and adapt
"""


import re

# Patterns that indicate the answer is code/patch output (not a factual answer)
_CODE_PATTERNS = re.compile(
    r"^(diff |---|\+\+\+|@@|patch|```)", re.MULTILINE
)


def _ensure_final_answer_tags(answer: str) -> str:
    """Wrap answer in FINAL_ANSWER tags if not already present and not code output."""
    if "<FINAL_ANSWER>" in answer.upper():
        return answer
    # Don't wrap code patches/diffs
    if _CODE_PATTERNS.search(answer):
        return answer
    return f"<FINAL_ANSWER>{answer}</FINAL_ANSWER>"


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def make_openai_client() -> AsyncOpenAI:
    """Create an OpenAI client, using Azure when configured."""
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


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """General-purpose agent with a single agentic loop.

    Uses the OpenAI Responses API with shell access and a done tool.
    No classification, no skill routing — the model handles everything.

    Prompt design per OpenAI GPT-5.x guidance:
    - Developer role message (static, first) for prompt caching
    - Outcome-first: defines success criteria, not step-by-step process
    - Reasoning effort: "high" for complex agentic tasks
    - Context compaction: automatic via context_management parameter
    """

    def __init__(self):
        self.client = make_openai_client()
        self.model = os.environ.get("AGENT_MODEL", "gpt-5.4")
        self.step_limit = int(os.environ.get("AGENT_STEP_LIMIT", str(DEFAULT_STEP_LIMIT)))
        self.tool_result_limit = int(
            os.environ.get("AGENT_TOOL_RESULT_LIMIT", str(DEFAULT_TOOL_RESULT_LIMIT))
        )
        self.compact_threshold = int(
            os.environ.get("AGENT_COMPACT_THRESHOLD", str(DEFAULT_COMPACT_THRESHOLD))
        )

    async def execute(self, input_text: str) -> str:
        """Execute the task described in the input text. Returns the final answer."""
        logger.info("Received task (%d chars)", len(input_text))
        logger.debug("Task content: %s", input_text[:500])
        return await self._execute(input_text)

    async def passthrough(self, input_text: str) -> str:
        """Direct LLM call without developer message or tools.

        Used for self-contained prompts that already include full instructions
        (e.g. code generation benchmarks). No shell access, no FINAL_ANSWER wrapping.
        Lower reasoning effort for faster response.
        """
        logger.info("Passthrough mode (%d chars)", len(input_text))
        reasoning = is_reasoning_model(self.model)

        api_kwargs: dict = {
            "model": self.model,
            "input": [
                {"role": "developer", "content": (
                    "Follow all instructions in the user prompt precisely. "
                    "Use only the allowed output types/formats specified. "
                    "Generate minimal, correct code without over-engineering. "
                    "When working with graphs: use copy.deepcopy() for graph copies; "
                    "maintain hierarchy constraints (e.g. if a container node type "
                    "must contain children per the schema, add default children); "
                    "when removing a node, also remove all its descendants "
                    "(nodes reachable via outgoing containment edges) to prevent "
                    "orphaned/isolated nodes; "
                    "match the attribute formats of existing nodes in the graph."
                )},
                {"role": "user", "content": input_text},
            ],
            "store": False,
        }
        if reasoning:
            api_kwargs["reasoning"] = {"effort": "high", "summary": "auto"}
            api_kwargs["max_output_tokens"] = 16_000
        else:
            api_kwargs["temperature"] = 0.0
            api_kwargs["max_output_tokens"] = 8192

        try:
            response = await self.client.responses.create(**api_kwargs)
        except Exception as exc:
            logger.warning("Passthrough API error: %s", exc)
            return f"Error: {exc}"

        tracker.record(response, label="passthrough")
        tracker.log_summary()

        # Extract text from response
        text_parts = []
        for item in response.output:
            if hasattr(item, "type") and item.type == "message":
                for content in getattr(item, "content", []):
                    if hasattr(content, "text"):
                        text_parts.append(content.text)
            elif hasattr(item, "text"):
                text_parts.append(item.text)
        result = "\n".join(text_parts) if text_parts else ""
        logger.debug("Passthrough response:\n%s", result[:2000])
        return result

    # ------------------------------------------------------------------
    # Core agentic loop
    # ------------------------------------------------------------------

    async def _execute(self, input_text: str) -> str:
        """General-purpose agentic loop.

        Uses OpenAI Responses API with:
        - Developer role message (static, first position for prompt caching)
        - Shell tool (reasoning models) or run_command function (non-reasoning)
        - Done tool to signal completion
        - Context compaction for long-running tasks
        - Reasoning effort "high" for deep problem solving
        """
        reasoning = is_reasoning_model(self.model)
        tools = [SHELL_TOOL, DONE_TOOL] if reasoning else [RUN_COMMAND_TOOL, DONE_TOOL]

        # Input array: developer message first (static, cached), user message last (variable)
        # This ordering maximizes prompt cache hit rate per OpenAI guidance.
        items: list = [
            {"role": "developer", "content": DEVELOPER_MESSAGE},
            {"role": "user", "content": input_text},
        ]
        done_answer: str | None = None

        for step in range(self.step_limit):
            logger.info("Step %d/%d", step + 1, self.step_limit)

            # Inject step-budget warning when running low
            step_warning = None
            remaining = self.step_limit - step
            if remaining == 5:
                step_warning = (
                    "[SYSTEM: You have 5 steps remaining. Start wrapping up. "
                    "If you have a partial answer or best estimate, call `done` now. "
                    "Wrap factual answers in <FINAL_ANSWER> tags.]"
                )
            elif remaining <= 2:
                step_warning = (
                    "[SYSTEM: FINAL STEP. You MUST call `done` RIGHT NOW with "
                    "your best answer or estimate. Any answer is better than none. "
                    "Use <FINAL_ANSWER> tags for factual answers.]"
                )

            api_kwargs: dict = {
                "model": self.model,
                "input": items if not step_warning else items + [
                    {"role": "developer", "content": step_warning}
                ],
                "tools": tools,
                "parallel_tool_calls": False,
                "store": False,
            }
            if reasoning:
                # GPT-5.x reasoning model settings per OpenAI best practices:
                # - include encrypted reasoning for stateless multi-turn
                # - compaction to manage long contexts
                # - high effort for complex agentic tasks
                # - summary for debugging/observability
                api_kwargs["include"] = ["reasoning.encrypted_content"]
                api_kwargs["context_management"] = [
                    {"type": "compaction", "compact_threshold": self.compact_threshold},
                ]
                api_kwargs["reasoning"] = {"effort": "high", "summary": "auto"}
                api_kwargs["max_output_tokens"] = 16_000
            else:
                api_kwargs["temperature"] = 0.0
                api_kwargs["max_output_tokens"] = 4096

            try:
                response = await self.client.responses.create(**api_kwargs)
            except Exception as exc:
                logger.warning("API error at step %d: %s", step, exc)
                break

            tracker.record(response, label=f"step={step+1}")

            # Append output and handle compaction
            items.extend(response.output)
            last_compaction_idx = None
            for i, it in enumerate(items):
                if hasattr(it, "type") and it.type == "compaction":
                    last_compaction_idx = i
            if last_compaction_idx is not None and last_compaction_idx > 0:
                items = items[last_compaction_idx:]

            # Collect tool calls
            shell_calls = [it for it in response.output if it.type == "shell_call"]
            function_calls = [it for it in response.output if it.type == "function_call"]

            # No tool calls = model gave a direct text response
            if not shell_calls and not function_calls:
                text_parts = []
                for item in response.output:
                    if hasattr(item, "type") and item.type == "message":
                        for content in getattr(item, "content", []):
                            if hasattr(content, "text"):
                                text_parts.append(content.text)
                    elif hasattr(item, "text"):
                        text_parts.append(item.text)
                if text_parts:
                    return _ensure_final_answer_tags("\n".join(text_parts))
                break

            # Process shell_call items (reasoning models)
            for sc in shell_calls:
                commands = []
                if hasattr(sc, "action") and sc.action:
                    commands = (
                        sc.action.commands if hasattr(sc.action, "commands") else []
                    )
                if not commands:
                    commands = ["echo '(no command)'"]

                logger.info("[step %d] shell: %s", step + 1, commands[0][:120])

                results = []
                for cmd in commands:
                    r = await run_shell_command(cmd)
                    results.append({
                        "stdout": truncate(r["stdout"], self.tool_result_limit),
                        "stderr": truncate(r["stderr"], self.tool_result_limit),
                        "outcome": {"type": "exit", "exit_code": r["exit_code"]},
                    })

                max_output_length = self.tool_result_limit
                if (
                    hasattr(sc, "action")
                    and hasattr(sc.action, "max_output_length")
                    and sc.action.max_output_length
                ):
                    max_output_length = sc.action.max_output_length

                items.append({
                    "type": "shell_call_output",
                    "call_id": sc.call_id,
                    "output": results,
                    "max_output_length": max_output_length,
                })

            # Process function_call items (non-reasoning models + done tool)
            for fc in function_calls:
                name = fc.name
                try:
                    args = json.loads(fc.arguments)
                except json.JSONDecodeError:
                    args = {}

                if name == "done":
                    done_answer = args.get("answer", "")
                    items.append({
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": "Task completed.",
                    })
                elif name == "run_command":
                    cmd = args.get("command", "echo 'no command'")
                    logger.info("[step %d] cmd: %s", step + 1, cmd[:120])
                    r = await run_shell_command(cmd)
                    output = r["stdout"]
                    if r["stderr"]:
                        output += f"\nSTDERR: {r['stderr']}"
                    output += f"\n[exit code: {r['exit_code']}]"
                    items.append({
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": truncate(output, self.tool_result_limit),
                    })

            if done_answer is not None:
                tracker.log_summary()
                return _ensure_final_answer_tags(done_answer)

        # Exhausted step budget — return whatever we have
        if done_answer is not None:
            tracker.log_summary()
            return _ensure_final_answer_tags(done_answer)

        for item in reversed(items):
            if hasattr(item, "type") and item.type == "message":
                for content in getattr(item, "content", []):
                    if hasattr(content, "text"):
                        return _ensure_final_answer_tags(content.text)
            elif isinstance(item, dict) and item.get("type") == "function_call_output":
                continue
            elif hasattr(item, "text"):
                return _ensure_final_answer_tags(item.text)

        # Best-effort: extract answer from recent tool outputs
        best_effort = self._extract_best_effort(items)
        if best_effort:
            logger.info("Best-effort extraction: %s", best_effort[:100])
            return _ensure_final_answer_tags(best_effort)

        tracker.log_summary()
        return "<FINAL_ANSWER>Unable to determine answer within step budget</FINAL_ANSWER>"

    @staticmethod
    def _extract_best_effort(items: list) -> str | None:
        """Scan recent tool outputs for a plausible answer.

        Looks for the last substantive text output from tool calls that
        might contain the answer the model was converging on.
        """
        # Look at recent function_call_output or shell_call_output items
        for item in reversed(items[-20:]):
            if isinstance(item, dict) and item.get("type") == "function_call_output":
                output = item.get("output", "")
                if output and output != "Task completed." and len(output) > 5:
                    # Skip error-only outputs
                    if output.startswith("Error:") or output.startswith("Command timed out"):
                        continue
                    # Return first non-trivial output (likely last computation result)
                    lines = output.strip().splitlines()
                    # Look for short outputs (likely answers) vs long outputs (data dumps)
                    if len(lines) <= 5 and len(output) < 500:
                        return output.strip()
            elif isinstance(item, dict) and item.get("type") == "shell_call_output":
                results = item.get("output", [])
                for r in reversed(results):
                    stdout = r.get("stdout", "")
                    if stdout and len(stdout.strip()) < 500:
                        lines = stdout.strip().splitlines()
                        if len(lines) <= 5 and lines:
                            return stdout.strip()
        return None
