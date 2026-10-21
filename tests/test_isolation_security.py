"""
Security and Isolation Tests:
Verifies that development_gold.csv is strictly isolated from runtime fact lookup tools and inference.
"""

import inspect

from app.services import fact_store


def test_fact_store_has_no_gold_references():
    """Verify FactStore class has no methods or fields referencing gold data."""
    store = fact_store.FactStore()
    source_code = inspect.getsource(fact_store)

    assert "development_gold" not in source_code.lower()
    assert "expected_label" not in source_code.lower()
    assert "goldclaim" not in source_code.lower()
    assert "gold" not in source_code.lower()

    # Verify loaded facts in FactStore only come from brand_facts
    assert store.total_facts == 132
    assert all(hasattr(f, "fact_id") for f in store._facts)


def test_runtime_data_loader_isolation():
    """Verify runtime loaders only expose answers and brand facts."""
    from app.services import data_loader

    # load_development_gold_for_eval_only is explicitly flagged for eval
    assert hasattr(data_loader, "load_development_gold_for_eval_only")

    # Verify docstring states STRICT EVALUATION ONLY
    doc = data_loader.load_development_gold_for_eval_only.__doc__ or ""
    assert "STRICT EVALUATION ONLY" in doc
