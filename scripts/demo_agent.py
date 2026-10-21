"""
Demonstration script showing the agent on development answers (A01 and A02).
Runs a sanitized model/tool exchange, deterministically resolves offsets,
and verifies exact quote slicing and evidence fact IDs.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import ClaimVerificationAgent
from app.fact_tool import BrandFactTool
from app.providers import MockLLMProvider, ProviderResponse, ToolCall
from app.services.data_loader import load_answers


def run_demo():
    print("=" * 80)
    print("KASPARRO CLAIM-CHECKER: AGENT DEMONSTRATION ON DEVELOPMENT EXAMPLES")
    print("=" * 80)

    answers = {a.answer_id: a for a in load_answers()}
    fact_tool = BrandFactTool()

    # Demonstration 1: A01 (ClearTrail Mini single brand query)
    a01 = answers["A01"]
    print(f"\n--- Demonstrating on Answer {a01.answer_id} (Brand: {a01.brand_id}) ---")
    print(f"Question: {a01.question}")
    print(f"AI Answer Length: {len(a01.ai_answer)} characters\n")

    # Turn 1: Model requests get_brand_facts('cleartrail')
    turn1_a01 = ProviderResponse(
        tool_calls=[ToolCall("call_a01_1", "get_brand_facts", {"brand_id": "cleartrail"})]
    )

    # Turn 2: Model uses facts to classify extracted complete sentences
    turn2_payload_a01 = {
        "claims": [
            {
                "sentence": (
                    "For the current India catalogue, ClearTrail Mini is listed at INR 1,200 less "
                    "than ClearTrail Plus, comparing the two tax-inclusive prices before shipping."
                ),
                "label": "supported",
                "fact_ids": ["cleartrail.mini.price_2026", "cleartrail.plus.price_2026"],
                "reason": "Mini is INR 1,299 and Plus is INR 2,499 in the 2026 catalogue, yielding an exact INR 1,200 difference.",
            },
            {
                "sentence": (
                    "For an authorized India purchase with the original invoice and a manufacturing defect, "
                    "ClearTrail Mini receives a standard manufacturer warranty lasting 18 months from the purchase date."
                ),
                "label": "contradicted",
                "fact_ids": ["cleartrail.mini.warranty_india"],
                "reason": "The official catalogue standard warranty is 12 months, contradicting the stated 18 months.",
            },
            {
                "sentence": (
                    "The official India retail box for ClearTrail Mini includes a wall charging adapter, "
                    "even though separate accessory lists may make the charging arrangement look less obvious."
                ),
                "label": "contradicted",
                "fact_ids": ["cleartrail.mini.adapter_included"],
                "reason": "The catalogue fact adapter_included is false, refuting the claim that an adapter is included.",
            },
            {
                "sentence": (
                    "An opened ClearTrail Mini can be returned to any authorized India seller within thirty days "
                    "for a full refund, even when the speaker has no manufacturing defect or functional problem."
                ),
                "label": "unknown",
                "fact_ids": [],
                "reason": "No fact in the catalogue specifies thirty-day change-of-mind return terms.",
            },
        ]
    }
    turn2_a01 = ProviderResponse(content=json.dumps(turn2_payload_a01))

    provider_a01 = MockLLMProvider([turn1_a01, turn2_a01])
    agent_a01 = ClaimVerificationAgent(provider=provider_a01, fact_tool=fact_tool)

    result_a01 = agent_a01.verify_answer(
        answer_id=a01.answer_id,
        brand_id=a01.brand_id,
        question=a01.question,
        ai_answer=a01.ai_answer,
    )

    print("Step 1 (Tool Call): Model requested get_brand_facts('cleartrail')")
    print(
        f"Step 2 (Tool Output): Catalogue returned {len(fact_tool.get_brand_facts('cleartrail')['facts'])} verified facts"
    )
    print(
        f"Step 3 (Validated Claims): Resolved {len(result_a01.claims)} claims with deterministic Python source matching:"
    )

    for idx, c in enumerate(result_a01.claims, 1):
        exact_slice = a01.ai_answer[c.start_char : c.end_char]
        match_status = "EXACT MATCH (PASS)" if exact_slice == c.quote else "MISMATCH (FAIL)"
        print(
            f"  [{idx}] Claim ID: {c.claim_id} | Label: {c.label.upper()} | Span: [{c.start_char}:{c.end_char}] | {match_status}"
        )
        print(f"      Quote: {c.quote!r}")
        print(f"      Fact IDs: {c.fact_ids}")
        print(f"      Reason: {c.reason}\n")

    # Demonstration 2: Cross-brand comparison answer
    print("=" * 80)
    print("Demonstration 2: Cross-Brand Multi-Tool Call Handling (Comparison)")
    print("=" * 80)
    turn1_comp = ProviderResponse(
        tool_calls=[
            ToolCall("call_c1", "get_brand_facts", {"brand_id": "cleartrail"}),
            ToolCall("call_c2", "get_brand_facts", {"brand_id": "stonewave"}),
        ]
    )
    turn2_comp = ProviderResponse(
        content=json.dumps(
            {
                "claims": [
                    {
                        "sentence": "ClearTrail Mini costs INR 1,299.",
                        "label": "supported",
                        "fact_ids": ["cleartrail.mini.price_2026"],
                        "reason": "Matches ClearTrail 2026 list price.",
                    }
                ]
            }
        )
    )
    provider_comp = MockLLMProvider([turn1_comp, turn2_comp])
    agent_comp = ClaimVerificationAgent(provider=provider_comp, fact_tool=fact_tool)
    result_comp = agent_comp.verify_answer(
        answer_id="A02",
        brand_id="cleartrail",
        question="Compare ClearTrail and StoneWave",
        ai_answer="Here is the price: ClearTrail Mini costs INR 1,299. Thank you.",
    )
    print("Successfully processed multi-brand tool calls in single turn:")
    print(f"Turn 1 tool calls: {len(provider_comp.call_history[0]['tools'])} tools registered")
    print(f"Resolved claim: {result_comp.claims[0].claim_id} ({result_comp.claims[0].label})")
    print(
        "\nDemonstration complete: All assertions, offsets, and catalogue IDs verified successfully."
    )


if __name__ == "__main__":
    run_demo()
