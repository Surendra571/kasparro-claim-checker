"""
Domain models for answers, facts, claims, and verification.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VerificationLabel(str, Enum):
    """The 3 valid verification labels according to the assessment spec."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class SplitType(str, Enum):
    """Split types present in data and specifications."""

    DEVELOPMENT = "development"
    CHECK = "check"
    FINAL_CHECK = "final-check"

    @classmethod
    def normalize(cls, value: str) -> "SplitType":
        val = value.strip().lower()
        if val in ("check", "final-check"):
            return cls.CHECK
        if val == "development":
            return cls.DEVELOPMENT
        raise ValueError(f"Unknown split: {value}")


class AnswerRecord(BaseModel):
    """Represents a row in answers.csv."""

    model_config = ConfigDict(frozen=True)

    answer_id: str
    split: SplitType
    brand_id: str
    question: str
    ai_answer: str

    @field_validator("split", mode="before")
    @classmethod
    def parse_split(cls, v: str) -> SplitType:
        if isinstance(v, SplitType):
            return v
        return SplitType.normalize(str(v))


class BrandFact(BaseModel):
    """Represents a row in brand_facts.csv."""

    model_config = ConfigDict(frozen=True)

    fact_id: str
    brand_id: str
    model_id: str
    field: str
    value: str
    value_type: str
    unit: str = ""
    condition: str = ""
    valid_from: str
    valid_to: str

    @property
    def parsed_value(self):
        """Converts value based on value_type."""
        if self.value_type == "boolean":
            return self.value.strip().lower() == "true"
        elif self.value_type == "number":
            try:
                return float(self.value) if "." in self.value else int(self.value)
            except ValueError:
                return self.value
        return self.value


class Claim(BaseModel):
    """An extracted claim bound to character offsets in the original ai_answer."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    answer_id: str
    start_char: int
    end_char: int
    quote: str


class GoldClaim(BaseModel):
    """A benchmark claim from development_gold.csv."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    answer_id: str
    start_char: int
    end_char: int
    quote: str
    expected_label: VerificationLabel
    evidence_fact_ids: list[str] = Field(default_factory=list)
    explanation: str = ""
    difficulty_tags: list[str] = Field(default_factory=list)

    @field_validator("expected_label", mode="before")
    @classmethod
    def parse_label(cls, v: str) -> VerificationLabel:
        if isinstance(v, VerificationLabel):
            return v
        return VerificationLabel(str(v).strip().lower())

    @field_validator("evidence_fact_ids", mode="before")
    @classmethod
    def parse_evidence(cls, v) -> list[str]:
        if isinstance(v, list):
            return v
        if not v or not str(v).strip():
            return []
        return [part.strip() for part in str(v).split("|") if part.strip()]

    @field_validator("difficulty_tags", mode="before")
    @classmethod
    def parse_tags(cls, v) -> list[str]:
        if isinstance(v, list):
            return v
        if not v or not str(v).strip():
            return []
        return [tag.strip() for tag in str(v).split("|") if tag.strip()]


class VerificationResult(BaseModel):
    """Output prediction for an extracted claim."""

    claim_id: str
    label: VerificationLabel
    evidence_fact_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    reasoning: str = ""
