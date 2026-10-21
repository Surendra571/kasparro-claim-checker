"""
Validation and Deterministic Python Source Matching.
Resolves exact character boundaries in ai_answer without relying on model offset guesses,
validates fact IDs against the catalogue, enforces label schemas, and ensures failure preservation.
"""

import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

VALID_LABELS = {"supported", "contradicted", "unknown"}


class ValidationError(Exception):
    """Base exception for claim validation errors."""


class UnresolvedSpanError(ValidationError):
    """Raised when a claim sentence cannot be located in ai_answer."""


class AmbiguousSpanError(ValidationError):
    """Raised when a claim sentence appears multiple times and cannot be deterministically resolved."""


class InvalidFactIdError(ValidationError):
    """Raised when an uncatalogued fact ID is cited."""


class InvalidLabelError(ValidationError):
    """Raised when a label is not supported, contradicted, or unknown."""


class InvalidReasonError(ValidationError):
    """Raised when reason exceeds the 2-sentence maximum."""


class OverlappingClaimsError(ValidationError):
    """Raised when extracted claim spans overlap."""


class ValidatedClaim(BaseModel):
    """A fully validated claim with deterministic character boundaries."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: str
    fact_ids: list[str] = Field(default_factory=list)
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "quote": self.quote,
            "label": self.label,
            "fact_ids": self.fact_ids,
            "reason": self.reason,
        }


class AnswerVerificationResult(BaseModel):
    """The complete verified result for an answer."""

    model_config = ConfigDict(frozen=True)

    answer_id: str
    claims: list[ValidatedClaim]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_id": self.answer_id,
            "claims": [c.to_dict() for c in self.claims],
        }


def count_sentences(text: str) -> int:
    """
    Counts sentences in a text string.
    Splits by standard punctuation (. ! ?) followed by whitespace or end of string.
    """
    cleaned = text.strip()
    if not cleaned:
        return 0
    # Split on sentence-ending punctuation followed by whitespace
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", cleaned) if p.strip()]
    return len(parts)


def find_all_occurrences(needle: str, haystack: str) -> list[int]:
    """Finds all start indices of needle in haystack."""
    indices = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        indices.append(idx)
        start = idx + 1
    return indices


def resolve_sentence_span(
    sentence: str,
    ai_answer: str,
    search_cursor: int = 0,
) -> tuple[int, int, str]:
    """
    Deterministically locates sentence in ai_answer.
    Uses sequential search cursor to resolve repeated sentences.
    Rejects ambiguous or unresolvable sentences without guessing.

    Returns:
        (start_char, end_char, exact_quote)
    """
    cleaned_sent = sentence.strip()
    if not cleaned_sent:
        raise UnresolvedSpanError("Empty claim sentence provided.")

    # Try exact match as provided
    candidates = [cleaned_sent]

    # Also try stripped of outer quotation marks if model wrapped it
    stripped_quotes = cleaned_sent.strip("\"'“”‘’")
    if stripped_quotes and stripped_quotes != cleaned_sent:
        candidates.append(stripped_quotes)

    resolved_match: tuple[int, int, str] | None = None

    for cand in candidates:
        all_indices = find_all_occurrences(cand, ai_answer)
        if not all_indices:
            continue

        if len(all_indices) == 1:
            start_idx = all_indices[0]
            end_idx = start_idx + len(cand)
            resolved_match = (start_idx, end_idx, cand)
            break

        # Multiple occurrences: use sequential search_cursor context
        post_cursor = [idx for idx in all_indices if idx >= search_cursor]
        if post_cursor:
            # Deterministically take the next sequential occurrence after cursor
            start_idx = post_cursor[0]
            end_idx = start_idx + len(cand)
            resolved_match = (start_idx, end_idx, cand)
            break
        else:
            # All occurrences occurred before search cursor (out of sequence / duplicate)
            raise AmbiguousSpanError(
                f"Sentence appeared {len(all_indices)} time(s) but all were before cursor {search_cursor}: {cand!r}"
            )

    if resolved_match is None:
        raise UnresolvedSpanError(f"Could not locate sentence in source ai_answer: {sentence!r}")

    start_char, end_char, quote = resolved_match
    # Strict validation: slice must match exactly
    assert ai_answer[start_char:end_char] == quote, "Deterministic slice mismatch!"
    return start_char, end_char, quote


def validate_and_build_claims(
    raw_claims: Sequence[dict[str, Any]],
    ai_answer: str,
    answer_id: str,
    valid_fact_ids: set[str],
) -> list[ValidatedClaim]:
    """
    Validates raw claims from model output:
    1. Resolves exact character boundaries deterministically.
    2. Enforces valid labels: 'supported', 'contradicted', 'unknown'.
    3. Validates that every fact ID exists in the catalogue.
    4. Enforces reason length <= 2 sentences.
    5. Deterministically assigns claim_id as answer_id:start_char:end_char.
    6. Sorts by start_char.
    7. Ensures no overlapping claims.
    """
    if not isinstance(raw_claims, (list, tuple)):
        raise ValidationError(f"raw_claims must be a list, got {type(raw_claims).__name__}")

    validated_claims: list[ValidatedClaim] = []
    cursor = 0

    for idx, raw in enumerate(raw_claims):
        if not isinstance(raw, dict):
            raise ValidationError(f"Claim at index {idx} must be a dict, got {type(raw).__name__}")

        # Extract sentence / quote
        raw_sentence = raw.get("sentence") or raw.get("quote")
        if not raw_sentence or not isinstance(raw_sentence, str):
            raise ValidationError(f"Claim at index {idx} missing valid 'sentence' text.")

        # Extract & validate label
        raw_label = str(raw.get("label", "")).strip().lower()
        if raw_label not in VALID_LABELS:
            raise InvalidLabelError(
                f"Invalid label {raw.get('label')!r} at claim index {idx}. Must be one of {sorted(VALID_LABELS)}."
            )

        # Extract & validate fact IDs
        raw_fids = raw.get("fact_ids", [])
        if isinstance(raw_fids, str):
            fids = [f.strip() for f in raw_fids.split("|") if f.strip()]
        elif isinstance(raw_fids, list):
            fids = [str(f).strip() for f in raw_fids if str(f).strip()]
        else:
            raise ValidationError(f"fact_ids at claim index {idx} must be a list or string.")

        for fid in fids:
            if fid not in valid_fact_ids:
                raise InvalidFactIdError(f"Claim index {idx} cites uncatalogued fact ID: {fid!r}")

        # Extract & validate reason
        reason = str(raw.get("reason", "")).strip()
        num_sentences = count_sentences(reason)
        if num_sentences > 2:
            raise InvalidReasonError(
                f"Reason at claim index {idx} exceeds 2 sentences ({num_sentences} sentences found): {reason!r}"
            )

        # Resolve exact span
        start_char, end_char, quote = resolve_sentence_span(
            sentence=raw_sentence,
            ai_answer=ai_answer,
            search_cursor=cursor,
        )

        cursor = end_char
        claim_id = f"{answer_id}:{start_char}:{end_char}"

        validated_claims.append(
            ValidatedClaim(
                claim_id=claim_id,
                start_char=start_char,
                end_char=end_char,
                quote=quote,
                label=raw_label,
                fact_ids=fids,
                reason=reason,
            )
        )

    # Sort deterministically by start_char
    validated_claims.sort(key=lambda c: c.start_char)

    # Check for overlapping spans
    for i in range(len(validated_claims) - 1):
        c1 = validated_claims[i]
        c2 = validated_claims[i + 1]
        if c1.end_char > c2.start_char:
            raise OverlappingClaimsError(
                f"Overlapping claims detected: {c1.claim_id} ({c1.start_char}:{c1.end_char}) "
                f"and {c2.claim_id} ({c2.start_char}:{c2.end_char})"
            )

    return validated_claims
