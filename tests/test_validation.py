"""
Tests for validation and deterministic Python source matching.
"""

import pytest

from app.validation import (
    AmbiguousSpanError,
    InvalidFactIdError,
    InvalidLabelError,
    InvalidReasonError,
    OverlappingClaimsError,
    UnresolvedSpanError,
    count_sentences,
    resolve_sentence_span,
    validate_and_build_claims,
)


def test_count_sentences():
    """Verify sentence counting logic."""
    assert count_sentences("This is one sentence.") == 1
    assert count_sentences("First sentence. Second sentence!") == 2
    assert count_sentences("One? Two! Three.") == 3
    assert count_sentences("") == 0


def test_resolve_sentence_span_single_exact_match():
    """Verify unambiguous single sentence match finds exact character offsets."""
    text = "Hello world. The ClearTrail Mini costs INR 1,299. Thank you."
    sent = "The ClearTrail Mini costs INR 1,299."

    start, end, quote = resolve_sentence_span(sent, text)
    assert text[start:end] == sent
    assert quote == sent
    assert start == 13
    assert end == 49


def test_resolve_sentence_span_repeated_with_cursor():
    """Verify repeated sentences are disambiguated by cursor position."""
    text = "ClearTrail is great. Some other text. ClearTrail is great. End."
    sent = "ClearTrail is great."

    # First occurrence
    s1, e1, _ = resolve_sentence_span(sent, text, search_cursor=0)
    assert s1 == 0
    assert e1 == 20

    # Second occurrence after cursor
    s2, e2, _ = resolve_sentence_span(sent, text, search_cursor=e1)
    assert s2 == 38
    assert e2 == 58


def test_resolve_sentence_span_unlocatable_rejected():
    """Verify hallucinated or altered sentence text is rejected without guessing."""
    text = "The battery lasts 12 hours under laboratory conditions."
    sent = "The battery lasts 10 hours under normal conditions."

    with pytest.raises(UnresolvedSpanError):
        resolve_sentence_span(sent, text)


def test_resolve_sentence_span_ambiguous_rejected():
    """Verify repeated sentence appearing only before cursor is rejected."""
    text = "Repeat sentence. Middle text. Repeat sentence. End."
    sent = "Repeat sentence."

    # Cursor is at the end of the text (past all occurrences)
    with pytest.raises(AmbiguousSpanError):
        resolve_sentence_span(sent, text, search_cursor=len(text))


def test_validate_and_build_claims_full_pipeline():
    """Verify complete validation pipeline assigns deterministic IDs and sorts by start_char."""
    ai_answer = (
        "Welcome to the overview. "
        "The ClearTrail Mini costs INR 1,299. "
        "The warranty lasts 12 months. "
        "Have a great day."
    )
    raw_claims = [
        {
            "sentence": "The warranty lasts 12 months.",
            "label": "supported",
            "fact_ids": ["cleartrail.mini.warranty_india"],
            "reason": "Confirmed by India catalogue warranty fact.",
        },
        {
            "sentence": "The ClearTrail Mini costs INR 1,299.",
            "label": "supported",
            "fact_ids": ["cleartrail.mini.price_2026"],
            "reason": "Matches 2026 catalogue price.",
        },
    ]

    valid_facts = {"cleartrail.mini.price_2026", "cleartrail.mini.warranty_india"}
    validated = validate_and_build_claims(
        raw_claims=raw_claims,
        ai_answer=ai_answer,
        answer_id="A01",
        valid_fact_ids=valid_facts,
    )

    assert len(validated) == 2
    # Must be sorted by start_char
    assert validated[0].start_char < validated[1].start_char
    assert validated[0].quote == "The ClearTrail Mini costs INR 1,299."
    assert validated[0].claim_id == f"A01:{validated[0].start_char}:{validated[0].end_char}"
    assert validated[1].quote == "The warranty lasts 12 months."


def test_validate_and_build_claims_rejects_invalid_fact_id():
    """Verify uncatalogued fact IDs are rejected."""
    ai_answer = "The ClearTrail Mini costs INR 1,299."
    raw_claims = [
        {
            "sentence": "The ClearTrail Mini costs INR 1,299.",
            "label": "supported",
            "fact_ids": ["cleartrail.mini.non_existent_fact"],
            "reason": "Looks good.",
        }
    ]

    with pytest.raises(InvalidFactIdError):
        validate_and_build_claims(
            raw_claims=raw_claims,
            ai_answer=ai_answer,
            answer_id="A01",
            valid_fact_ids={"cleartrail.mini.price_2026"},
        )


def test_validate_and_build_claims_rejects_invalid_label():
    """Verify illegal labels are rejected."""
    ai_answer = "The ClearTrail Mini costs INR 1,299."
    raw_claims = [
        {
            "sentence": "The ClearTrail Mini costs INR 1,299.",
            "label": "TRUE",  # Invalid label
            "fact_ids": ["cleartrail.mini.price_2026"],
            "reason": "Confirmed.",
        }
    ]

    with pytest.raises(InvalidLabelError):
        validate_and_build_claims(
            raw_claims=raw_claims,
            ai_answer=ai_answer,
            answer_id="A01",
            valid_fact_ids={"cleartrail.mini.price_2026"},
        )


def test_validate_and_build_claims_rejects_excessive_reason():
    """Verify reasons exceeding 2 sentences are rejected."""
    ai_answer = "The ClearTrail Mini costs INR 1,299."
    raw_claims = [
        {
            "sentence": "The ClearTrail Mini costs INR 1,299.",
            "label": "supported",
            "fact_ids": ["cleartrail.mini.price_2026"],
            "reason": "Sentence one. Sentence two! Sentence three should fail.",
        }
    ]

    with pytest.raises(InvalidReasonError):
        validate_and_build_claims(
            raw_claims=raw_claims,
            ai_answer=ai_answer,
            answer_id="A01",
            valid_fact_ids={"cleartrail.mini.price_2026"},
        )


def test_validate_and_build_claims_rejects_overlapping_spans():
    """Verify overlapping spans are rejected."""
    ai_answer = "The ClearTrail Mini costs INR 1,299 in India."
    raw_claims = [
        {
            "sentence": "The ClearTrail Mini costs INR 1,299",
            "label": "supported",
            "fact_ids": [],
            "reason": "First clause.",
        },
        {
            "sentence": "costs INR 1,299 in India.",
            "label": "supported",
            "fact_ids": [],
            "reason": "Overlapping clause.",
        },
    ]

    with pytest.raises(OverlappingClaimsError):
        validate_and_build_claims(
            raw_claims=raw_claims,
            ai_answer=ai_answer,
            answer_id="A01",
            valid_fact_ids=set(),
        )
