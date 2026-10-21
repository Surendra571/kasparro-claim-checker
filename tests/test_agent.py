"""
Tests for ClaimVerificationAgent using MockLLMProvider.
Clearly separates fake-provider unit tests from live API calls.
Includes a sanitized multi-turn model/tool exchange and demonstration on development answer A01.
"""

import json

import pytest

from app.agent import (
    AgentError,
    ClaimVerificationAgent,
    IllegalToolNameError,
    InvalidAgentOutputError,
    MaxTurnsExceededError,
)
from app.fact_tool import BrandFactTool
from app.providers import (
    MockLLMProvider,
    ProviderError,
    ProviderResponse,
    ToolCall,
)
from app.services.data_loader import load_answers


def test_agent_normal_two_turn_tool_exchange():
    """
    Sanitized example model/tool exchange:
    Turn 1: Model calls get_brand_facts('cleartrail').
    Turn 2: Model uses tool facts to classify claims and returns JSON.
    """
    ai_answer = (
        "Welcome! The ClearTrail Mini costs INR 1,299 in India. "
        "The standard warranty is 12 months from purchase date."
    )

    # Simulated Turn 1 response: Model requests catalogue tool
    turn1_response = ProviderResponse(
        content=None,
        tool_calls=[
            ToolCall(
                call_id="call_abc123",
                name="get_brand_facts",
                arguments={"brand_id": "cleartrail"},
            )
        ],
    )

    # Simulated Turn 2 response: Model produces classified claims
    turn2_payload = {
        "claims": [
            {
                "sentence": "The ClearTrail Mini costs INR 1,299 in India.",
                "label": "supported",
                "fact_ids": ["cleartrail.mini.price_2026"],
                "reason": "Matches the 2026 official India catalogue list price.",
            },
            {
                "sentence": "The standard warranty is 12 months from purchase date.",
                "label": "supported",
                "fact_ids": ["cleartrail.mini.warranty_india"],
                "reason": "Confirmed by India manufacturer warranty specification.",
            },
        ]
    }
    turn2_response = ProviderResponse(content=json.dumps(turn2_payload))

    provider = MockLLMProvider([turn1_response, turn2_response])
    agent = ClaimVerificationAgent(provider=provider)

    result = agent.verify_answer(
        answer_id="A01",
        brand_id="cleartrail",
        question="What is the price and warranty of ClearTrail Mini?",
        ai_answer=ai_answer,
    )

    assert result.answer_id == "A01"
    assert len(result.claims) == 2

    c1, c2 = result.claims
    assert c1.quote == "The ClearTrail Mini costs INR 1,299 in India."
    assert ai_answer[c1.start_char : c1.end_char] == c1.quote
    assert c1.claim_id == f"A01:{c1.start_char}:{c1.end_char}"
    assert c1.label == "supported"
    assert c1.fact_ids == ["cleartrail.mini.price_2026"]

    assert c2.quote == "The standard warranty is 12 months from purchase date."
    assert ai_answer[c2.start_char : c2.end_char] == c2.quote
    assert c2.claim_id == f"A01:{c2.start_char}:{c2.end_char}"
    assert c2.label == "supported"


def test_agent_multiple_tool_calls_in_one_turn():
    """Verify model can request facts for multiple brands in a single comparison turn."""
    ai_answer = "Comparing both: ClearTrail Mini costs INR 1,299. StoneWave Go costs INR 1,999."

    turn1_response = ProviderResponse(
        content=None,
        tool_calls=[
            ToolCall(
                call_id="call_1",
                name="get_brand_facts",
                arguments={"brand_id": "cleartrail"},
            ),
            ToolCall(
                call_id="call_2",
                name="get_brand_facts",
                arguments={"brand_id": "stonewave"},
            ),
        ],
    )

    turn2_payload = {
        "claims": [
            {
                "sentence": "ClearTrail Mini costs INR 1,299.",
                "label": "supported",
                "fact_ids": ["cleartrail.mini.price_2026"],
                "reason": "Confirmed by ClearTrail catalogue.",
            },
            {
                "sentence": "StoneWave Go costs INR 1,999.",
                "label": "supported",
                "fact_ids": ["stonewave.go.price_2026"],
                "reason": "Confirmed by StoneWave catalogue.",
            },
        ]
    }
    turn2_response = ProviderResponse(content=json.dumps(turn2_payload))

    provider = MockLLMProvider([turn1_response, turn2_response])
    agent = ClaimVerificationAgent(provider=provider)

    result = agent.verify_answer(
        answer_id="A02",
        brand_id="cleartrail",
        question="Compare prices of Mini and Go.",
        ai_answer=ai_answer,
    )

    assert len(result.claims) == 2
    # Check that tool messages were appended for both calls
    history = provider.call_history[1]["messages"]
    tool_messages = [m for m in history if m.get("role") == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert tool_messages[1]["tool_call_id"] == "call_2"


def test_agent_max_turns_exceeded_raises_error():
    """Verify exceeding 3 turns raises MaxTurnsExceededError without converting to unknown."""
    provider = MockLLMProvider(
        [
            ProviderResponse(
                tool_calls=[ToolCall("1", "get_brand_facts", {"brand_id": "cleartrail"})]
            ),
            ProviderResponse(
                tool_calls=[ToolCall("2", "get_brand_facts", {"brand_id": "cloudnote"})]
            ),
            ProviderResponse(
                tool_calls=[ToolCall("3", "get_brand_facts", {"brand_id": "stonewave"})]
            ),
        ]
    )
    agent = ClaimVerificationAgent(provider=provider, max_turns=3)

    with pytest.raises(MaxTurnsExceededError):
        agent.verify_answer(
            answer_id="A01",
            brand_id="cleartrail",
            question="Question?",
            ai_answer="Sample text.",
        )


def test_agent_rejects_illegal_tool_name():
    """Verify illegal tool names are rejected with IllegalToolNameError."""
    provider = MockLLMProvider(
        [ProviderResponse(tool_calls=[ToolCall("1", "execute_command", {"cmd": "dir"})])]
    )
    agent = ClaimVerificationAgent(provider=provider)

    with pytest.raises(IllegalToolNameError):
        agent.verify_answer(
            answer_id="A01",
            brand_id="cleartrail",
            question="Question?",
            ai_answer="Sample text.",
        )


def test_agent_preserves_provider_error_as_answer_level_error():
    """Verify provider failures are never converted into empty successes or unknown labels."""

    class FailingProvider(MockLLMProvider):
        def generate(self, *args, **kwargs):
            raise ProviderError("Network connection reset by peer")

    agent = ClaimVerificationAgent(provider=FailingProvider())

    with pytest.raises(AgentError) as exc_info:
        agent.verify_answer(
            answer_id="A01",
            brand_id="cleartrail",
            question="Question?",
            ai_answer="Sample text.",
        )
    assert "Provider failure" in str(exc_info.value)


def test_agent_preserves_invalid_json_as_answer_level_error():
    """Verify invalid JSON output is raised as an answer-level error."""
    provider = MockLLMProvider([ProviderResponse(content="Sorry, I cannot format this as JSON.")])
    agent = ClaimVerificationAgent(provider=provider)

    with pytest.raises(InvalidAgentOutputError):
        agent.verify_answer(
            answer_id="A01",
            brand_id="cleartrail",
            question="Question?",
            ai_answer="Sample text.",
        )


def test_agent_demonstration_on_development_a01():
    """
    Demonstration on actual development answer A01:
    Verifies that the agent extracts real sentences from data/answers.csv,
    resolves exact offsets, maps evidence IDs to brand_facts.csv, and checks ai_answer[start:end] == quote.
    """
    answers = {a.answer_id: a for a in load_answers()}
    a01 = answers["A01"]

    # Realistic model response for A01 claims
    turn1_response = ProviderResponse(
        tool_calls=[ToolCall("call_1", "get_brand_facts", {"brand_id": "cleartrail"})]
    )

    turn2_payload = {
        "claims": [
            {
                "sentence": (
                    "For the current India catalogue, ClearTrail Mini is listed at INR 1,200 less "
                    "than ClearTrail Plus, comparing the two tax-inclusive prices before shipping."
                ),
                "label": "supported",
                "fact_ids": ["cleartrail.mini.price_2026", "cleartrail.plus.price_2026"],
                "reason": "ClearTrail Mini is 1299 INR and ClearTrail Plus is 2499 INR, a difference of 1200 INR.",
            },
            {
                "sentence": (
                    "For an authorized India purchase with the original invoice and a manufacturing defect, "
                    "ClearTrail Mini receives a standard manufacturer warranty lasting 18 months from the purchase date."
                ),
                "label": "contradicted",
                "fact_ids": ["cleartrail.mini.warranty_india"],
                "reason": "The official catalogue specifies a 12-month warranty, not 18 months.",
            },
            {
                "sentence": (
                    "An opened ClearTrail Mini can be returned to any authorized India seller within thirty days "
                    "for a full refund, even when the speaker has no manufacturing defect or functional problem."
                ),
                "label": "unknown",
                "fact_ids": [],
                "reason": "The catalogue contains no facts regarding 30-day change-of-mind return policies.",
            },
        ]
    }
    turn2_response = ProviderResponse(content=json.dumps(turn2_payload))

    provider = MockLLMProvider([turn1_response, turn2_response])
    agent = ClaimVerificationAgent(provider=provider, fact_tool=BrandFactTool())

    result = agent.verify_answer(
        answer_id=a01.answer_id,
        brand_id=a01.brand_id,
        question=a01.question,
        ai_answer=a01.ai_answer,
    )

    assert result.answer_id == "A01"
    assert len(result.claims) == 3

    for claim in result.claims:
        # Crucial check: deterministic source match slice
        assert a01.ai_answer[claim.start_char : claim.end_char] == claim.quote
        # Check claim ID format
        assert claim.claim_id == f"A01:{claim.start_char}:{claim.end_char}"
        # Check evidence facts exist in catalogue
        for fid in claim.fact_ids:
            assert agent.fact_tool.is_valid_fact_id(fid)

    # Check specific offsets matching actual gold slice for A01
    assert result.claims[0].start_char == 306
    assert result.claims[0].end_char == 460
    assert result.claims[0].label == "supported"

    assert result.claims[1].start_char == 461
    assert result.claims[1].end_char == 647
    assert result.claims[1].label == "contradicted"

    assert result.claims[2].start_char == 649
    assert result.claims[2].end_char == 832
    assert result.claims[2].label == "unknown"
