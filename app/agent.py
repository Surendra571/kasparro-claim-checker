"""
Claim Extraction and Fact Verification Agent.
Implements multi-turn tool-calling loop (max 3 turns), strict tool validation,
deterministic claim span resolution, and answer-level error preservation.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.fact_tool import (
    GET_BRAND_FACTS_TOOL_SCHEMA,
    BrandFactTool,
    FactToolError,
    IllegalToolNameError,
)
from app.providers import (
    BaseLLMProvider,
    ProviderError,
    ProviderTimeoutError,
)
from app.validation import (
    AnswerVerificationResult,
    ValidationError,
    validate_and_build_claims,
)

logger = logging.getLogger(__name__)

PROMPT_FILE = Path("prompts/v1.txt")


class AgentError(Exception):
    """Base exception for agent errors."""


class MaxTurnsExceededError(AgentError):
    """Raised when agent exceeds maximum allowed turns per answer."""


class InvalidAgentOutputError(AgentError):
    """Raised when agent produces unparseable or schema-violating output."""


def load_system_prompt() -> str:
    """Loads system prompt from prompts/v1.txt."""
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"System prompt file not found at {PROMPT_FILE}")
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def extract_json_payload(text: str) -> dict[str, Any]:
    """
    Extracts JSON object from model output text.
    Handles optional markdown code fences ```json ... ```.
    """
    cleaned = text.strip()
    # Check for markdown code fences
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(fence_pattern, cleaned)
    if match:
        cleaned = match.group(1).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise InvalidAgentOutputError(
            f"Failed to parse model response as JSON: {exc}. Raw text: {text[:200]!r}"
        ) from exc

    if not isinstance(data, dict):
        raise InvalidAgentOutputError(
            f"Model output JSON must be a dict, got {type(data).__name__}"
        )
    return data


class ClaimVerificationAgent:
    """
    Orchestrates the extraction and verification of claims for an answer.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        fact_tool: BrandFactTool | None = None,
        max_turns: int = 3,
        timeout: float = 30.0,
        prompt_path: Path | str | None = None,
    ):
        self._provider = provider
        self._fact_tool = fact_tool or BrandFactTool()
        self._max_turns = max_turns
        self._timeout = timeout
        if prompt_path:
            p = Path(prompt_path)
            with open(p, "r", encoding="utf-8") as f:
                self._system_prompt = f.read().strip()
        else:
            self._system_prompt = load_system_prompt()

    @property
    def fact_tool(self) -> BrandFactTool:
        return self._fact_tool

    def verify_answer(
        self,
        answer_id: str,
        brand_id: str,
        question: str,
        ai_answer: str,
    ) -> AnswerVerificationResult:
        """
        Executes claim extraction and verification over an answer.

        Strict constraints:
        - Maximum 3 model turns.
        - Exactly one permitted tool: get_brand_facts.
        - Tool arguments validated; illegal tool names rejected.
        - Failures preserved as answer-level errors (never masked as unknown).
        """
        user_message_content = (
            f"Answer ID: {answer_id}\n"
            f"Primary Brand: {brand_id}\n"
            f"User Question: {question}\n\n"
            f"AI Answer to verify:\n"
            f'"""\n'
            f"{ai_answer}\n"
            f'"""\n'
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_message_content},
        ]

        tools = [GET_BRAND_FACTS_TOOL_SCHEMA]
        turns = 0

        while turns < self._max_turns:
            turns += 1
            logger.debug(
                "Executing model turn %d/%d for answer %s", turns, self._max_turns, answer_id
            )

            try:
                response = self._provider.generate(
                    messages=messages,
                    tools=tools,
                    timeout=self._timeout,
                )
            except ProviderTimeoutError:
                raise
            except ProviderError as e:
                # Requirement 14: preserve failure as answer-level error
                raise AgentError(f"Provider failure on turn {turns} for {answer_id}: {e}") from e

            # Check if model requested tool call(s)
            if response.tool_calls:
                # Format assistant message with tool calls for history
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
                messages.append(assistant_msg)

                # Execute all requested tool calls in this turn
                for tc in response.tool_calls:
                    # Validate tool name
                    if tc.name != "get_brand_facts":
                        raise IllegalToolNameError(
                            f"Model attempted illegal tool call: {tc.name!r}. Only 'get_brand_facts' is permitted."
                        )

                    logger.info("Executing tool %s with args: %s", tc.name, tc.arguments)
                    try:
                        tool_result = self._fact_tool.execute_tool(tc.name, tc.arguments)
                        tool_content = json.dumps(tool_result)
                    except (FactToolError, ValueError, TypeError) as err:
                        tool_content = json.dumps({"error": str(err)})

                    # Append tool response message
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.call_id,
                            "content": tool_content,
                        }
                    )

                # Proceed to next turn to allow model to read tool results
                continue

            # If no tool calls, model provided final text content
            if not response.content:
                raise InvalidAgentOutputError(
                    f"Model returned empty content on turn {turns} for {answer_id} without tool calls."
                )

            # Parse JSON payload
            payload = extract_json_payload(response.content)
            raw_claims = payload.get("claims")
            if raw_claims is None or not isinstance(raw_claims, list):
                raise InvalidAgentOutputError(
                    f"Model response JSON missing 'claims' list: {payload!r}"
                )

            # Validate and deterministically locate claim spans
            try:
                validated_claims = validate_and_build_claims(
                    raw_claims=raw_claims,
                    ai_answer=ai_answer,
                    answer_id=answer_id,
                    valid_fact_ids=self._fact_tool.valid_fact_ids,
                )
            except ValidationError as val_err:
                # Requirement 14: preserve failure as answer-level error
                raise AgentError(f"Claim validation failed for {answer_id}: {val_err}") from val_err

            return AnswerVerificationResult(
                answer_id=answer_id,
                claims=validated_claims,
            )

        # Exceeded maximum turns without completion
        raise MaxTurnsExceededError(
            f"Agent exceeded maximum allowed turns ({self._max_turns}) for answer {answer_id}."
        )
