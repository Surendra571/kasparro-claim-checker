"""
In-memory and indexed Fact Store for brand facts.
Provides querying by brand, model, field, and temporal validity.
STRICT ISOLATION: Has zero access to benchmark evaluation datasets or ground truth.
"""

from app.models.domain import BrandFact
from app.services.data_loader import load_brand_facts


class FactStore:
    """Knowledge store for verified brand facts."""

    def __init__(self, facts: list[BrandFact] | None = None):
        self._facts: list[BrandFact] = facts if facts is not None else load_brand_facts()
        self._by_id: dict[str, BrandFact] = {}
        self._by_brand: dict[str, list[BrandFact]] = {}
        self._by_brand_model: dict[str, list[BrandFact]] = {}
        self._by_field: dict[str, list[BrandFact]] = {}
        self._build_indices()

    def _build_indices(self) -> None:
        for f in self._facts:
            self._by_id[f.fact_id] = f
            self._by_brand.setdefault(f.brand_id, []).append(f)
            key = f"{f.brand_id}:{f.model_id}"
            self._by_brand_model.setdefault(key, []).append(f)
            self._by_field.setdefault(f.field, []).append(f)

    @property
    def total_facts(self) -> int:
        return len(self._facts)

    def get_by_id(self, fact_id: str) -> BrandFact | None:
        return self._by_id.get(fact_id)

    def get_by_brand(self, brand_id: str) -> list[BrandFact]:
        return list(self._by_brand.get(brand_id.lower().strip(), []))

    def get_by_brand_and_model(self, brand_id: str, model_id: str) -> list[BrandFact]:
        key = f"{brand_id.lower().strip()}:{model_id.lower().strip()}"
        return list(self._by_brand_model.get(key, []))

    def search_facts(
        self,
        brand_id: str | None = None,
        model_id: str | None = None,
        field: str | None = None,
        date_filter: str | None = None,
    ) -> list[BrandFact]:
        results = self._facts
        if brand_id:
            brand_norm = brand_id.lower().strip()
            results = [f for f in results if f.brand_id == brand_norm]
        if model_id:
            model_norm = model_id.lower().strip()
            results = [f for f in results if f.model_id == model_norm]
        if field:
            field_norm = field.lower().strip()
            results = [f for f in results if f.field == field_norm]
        if date_filter:
            # Check temporal interval: valid_from <= date_filter <= valid_to
            results = [f for f in results if f.valid_from <= date_filter <= f.valid_to]
        return results

    def list_brands(self) -> list[str]:
        return sorted(self._by_brand.keys())

    def list_models_for_brand(self, brand_id: str) -> list[str]:
        brand_norm = brand_id.lower().strip()
        models = {f.model_id for f in self._by_brand.get(brand_norm, [])}
        return sorted(models)
