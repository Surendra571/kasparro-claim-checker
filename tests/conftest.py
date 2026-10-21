"""
Pytest configuration and shared fixtures.
"""

import pytest

from app.services.data_loader import (
    load_answers,
    load_brand_facts,
    load_development_gold_for_eval_only,
)
from app.services.fact_store import FactStore


@pytest.fixture(scope="session")
def answers():
    """Loads raw answers without modification."""
    return load_answers()


@pytest.fixture(scope="session")
def brand_facts():
    """Loads brand facts."""
    return load_brand_facts()


@pytest.fixture(scope="session")
def gold_claims():
    """Loads gold claims strictly for test/eval fixture."""
    return load_development_gold_for_eval_only()


@pytest.fixture(scope="session")
def fact_store(brand_facts):
    """Initializes fact store with verified brand facts."""
    return FactStore(brand_facts)
