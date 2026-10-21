"""
Authoritative Brand Facts Tool.
Loads the official catalogue from brand_facts.csv, strictly validates value types,
and exposes exactly one read-only tool: get_brand_facts(brand_id).
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings

FACT_VERSION = "speaker-catalogue-v4"
PERMITTED_TOOL_NAME = "get_brand_facts"


class FactToolError(Exception):
    """Base exception for fact tool operations."""


class UnknownBrandError(FactToolError):
    """Raised when an unknown brand_id is requested."""


class IllegalToolNameError(FactToolError):
    """Raised when an illegal or unpermitted tool name is called."""


@dataclass(frozen=True)
class CatalogueFact:
    """Represents a single typed catalogue fact."""

    fact_id: str
    brand_id: str
    model_id: str
    field: str
    value: int | float | bool | str
    value_type: str
    unit: str
    condition: str
    valid_from: str
    valid_to: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "brand_id": self.brand_id,
            "model_id": self.model_id,
            "field": self.field,
            "value": self.value,
            "value_type": self.value_type,
            "unit": self.unit,
            "condition": self.condition,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        }


def parse_typed_value(raw_val: str, value_type: str) -> int | float | bool | str:
    """
    Explicitly interprets value_type as number, boolean or string.
    NEVER uses bool("false") which wrongly evaluates to True in Python.
    """
    cleaned = raw_val.strip()
    vt = value_type.strip().lower()

    if vt == "boolean":
        lower_val = cleaned.lower()
        if lower_val == "true":
            return True
        elif lower_val == "false":
            return False
        raise ValueError(f"Invalid boolean value string: {raw_val!r}")

    elif vt == "number":
        try:
            return float(cleaned) if "." in cleaned else int(cleaned)
        except ValueError as exc:
            raise ValueError(f"Invalid number value string: {raw_val!r}") from exc

    elif vt == "string":
        return cleaned

    raise ValueError(f"Unsupported value_type: {value_type!r}")


class BrandFactTool:
    """
    Authoritative read-only catalogue tool.
    Loads brand_facts.csv and serves get_brand_facts(brand_id).
    """

    def __init__(self, csv_path: Path | None = None):
        self._path = csv_path or settings.BRAND_FACTS_FILE
        self._facts: list[CatalogueFact] = []
        self._facts_by_brand: dict[str, list[CatalogueFact]] = {}
        self._known_brands: set[str] = set()
        self._fact_ids: set[str] = set()
        self._load_catalogue()

    def _load_catalogue(self) -> None:
        """Parses brand_facts.csv with UTF-8 decoding and type coercion."""
        with open(self._path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fact_id = row["fact_id"].strip()
                brand_id = row["brand_id"].strip().lower()
                model_id = row["model_id"].strip().lower()
                field = row["field"].strip()
                val_type = row["value_type"].strip().lower()
                parsed_val = parse_typed_value(row["value"], val_type)

                # Correctly handle blank unit and condition fields
                raw_unit = row.get("unit", "")
                unit = raw_unit.strip() if raw_unit else ""

                raw_condition = row.get("condition", "")
                condition = raw_condition.strip() if raw_condition else ""

                valid_from = row["valid_from"].strip()
                valid_to = row["valid_to"].strip()

                fact = CatalogueFact(
                    fact_id=fact_id,
                    brand_id=brand_id,
                    model_id=model_id,
                    field=field,
                    value=parsed_val,
                    value_type=val_type,
                    unit=unit,
                    condition=condition,
                    valid_from=valid_from,
                    valid_to=valid_to,
                )
                self._facts.append(fact)
                self._facts_by_brand.setdefault(brand_id, []).append(fact)
                self._known_brands.add(brand_id)
                self._fact_ids.add(fact_id)

    @property
    def known_brands(self) -> set[str]:
        """Returns the set of valid brand IDs."""
        return set(self._known_brands)

    @property
    def valid_fact_ids(self) -> set[str]:
        """Returns all valid fact IDs in the catalogue."""
        return set(self._fact_ids)

    def is_valid_fact_id(self, fact_id: str) -> bool:
        """Checks if a fact_id exists in the catalogue."""
        return fact_id in self._fact_ids

    def get_brand_facts(self, brand_id: str) -> dict[str, Any]:
        """
        Authoritative tool: get_brand_facts(brand_id).
        Returns only the requested brand's catalogue facts and fact_version="speaker-catalogue-v4".
        Rejects unknown brands.
        """
        if not isinstance(brand_id, str):
            raise UnknownBrandError(f"brand_id must be a string, got {type(brand_id).__name__}")

        norm_brand = brand_id.strip().lower()
        if norm_brand not in self._known_brands:
            raise UnknownBrandError(
                f"Unknown brand_id: {brand_id!r}. Permitted brands: {sorted(self._known_brands)}"
            )

        brand_facts = self._facts_by_brand.get(norm_brand, [])
        return {
            "brand_id": norm_brand,
            "fact_version": FACT_VERSION,
            "facts": [f.to_dict() for f in brand_facts],
        }

    def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Validates tool name and arguments.
        Rejects illegal tool names, arbitrary code execution, and unknown brands.
        """
        if tool_name != PERMITTED_TOOL_NAME:
            raise IllegalToolNameError(
                f"Illegal tool name: {tool_name!r}. The only permitted tool is {PERMITTED_TOOL_NAME!r}."
            )

        if not isinstance(arguments, dict) or "brand_id" not in arguments:
            raise FactToolError(f"Missing required argument 'brand_id' in arguments: {arguments!r}")

        brand_id = arguments["brand_id"]
        return self.get_brand_facts(brand_id)


# OpenAI Function / Tool Calling Schema
GET_BRAND_FACTS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": PERMITTED_TOOL_NAME,
        "description": (
            "Retrieves the authoritative speaker catalogue facts for a specified brand. "
            "Call this tool for every brand mentioned in the answer text, including comparison brands."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "brand_id": {
                    "type": "string",
                    "description": "The brand identifier to look up (e.g. 'cleartrail', 'cloudnote', 'stonewave').",
                }
            },
            "required": ["brand_id"],
            "additionalProperties": False,
        },
    },
}
