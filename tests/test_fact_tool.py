"""
Tests for BrandFactTool and catalogue parsing.
Verifies explicit value typing, boolean false safety, read-only constraints,
unknown brand rejection, and illegal tool name rejection.
"""

import pytest

from app.fact_tool import (
    FACT_VERSION,
    BrandFactTool,
    IllegalToolNameError,
    UnknownBrandError,
    parse_typed_value,
)


def test_parse_typed_value_boolean():
    """Verify boolean values: 'true' -> True, 'false' -> False (never bool('false'))."""
    assert parse_typed_value("true", "boolean") is True
    assert parse_typed_value("True", "boolean") is True
    assert parse_typed_value("false", "boolean") is False
    assert parse_typed_value("False", "boolean") is False

    # Verify that Python's bool("false") bug is not present
    val = parse_typed_value("false", "boolean")
    assert val is False
    assert val != bool("false")  # bool("false") is True in Python!

    with pytest.raises(ValueError):
        parse_typed_value("not_a_bool", "boolean")


def test_parse_typed_value_number():
    """Verify number values are parsed as int or float."""
    int_val = parse_typed_value("1299", "number")
    assert isinstance(int_val, int)
    assert int_val == 1299

    float_val = parse_typed_value("1.5", "number")
    assert isinstance(float_val, float)
    assert float_val == 1.5

    with pytest.raises(ValueError):
        parse_typed_value("invalid_num", "number")


def test_parse_typed_value_string():
    """Verify string values are stripped and preserved."""
    s = parse_typed_value("  ClearTrail Mini  ", "string")
    assert s == "ClearTrail Mini"


def test_catalogue_loading():
    """Verify authoritative catalogue is parsed completely from brand_facts.csv."""
    tool = BrandFactTool()
    assert tool.known_brands == {"cleartrail", "cloudnote", "stonewave"}
    assert len(tool.valid_fact_ids) == 132

    # Check that boolean facts in catalogue are real booleans
    cloudnote_facts = tool.get_brand_facts("cloudnote")["facts"]
    wifi_facts = [f for f in cloudnote_facts if f["field"] == "wifi"]
    assert len(wifi_facts) > 0
    for wf in wifi_facts:
        assert isinstance(wf["value"], bool)

    # Check that unit and condition are strings and handle blanks
    for f in cloudnote_facts:
        assert isinstance(f["unit"], str)
        assert isinstance(f["condition"], str)


def test_get_brand_facts_success():
    """Verify get_brand_facts returns facts and fact_version='speaker-catalogue-v4'."""
    tool = BrandFactTool()
    res = tool.get_brand_facts("cleartrail")

    assert res["brand_id"] == "cleartrail"
    assert res["fact_version"] == FACT_VERSION
    assert isinstance(res["facts"], list)
    assert len(res["facts"]) > 0

    # Ensure all facts belong to cleartrail
    assert all(f["brand_id"] == "cleartrail" for f in res["facts"])


def test_get_brand_facts_unknown_brand_rejected():
    """Verify unknown brands are rejected with UnknownBrandError."""
    tool = BrandFactTool()

    with pytest.raises(UnknownBrandError):
        tool.get_brand_facts("bose")

    with pytest.raises(UnknownBrandError):
        tool.get_brand_facts("sony")

    with pytest.raises(UnknownBrandError):
        tool.get_brand_facts("")


def test_execute_tool_legal_and_illegal():
    """Verify execute_tool permits only 'get_brand_facts' and rejects illegal tools."""
    tool = BrandFactTool()

    # Legal tool call
    res = tool.execute_tool("get_brand_facts", {"brand_id": "stonewave"})
    assert res["brand_id"] == "stonewave"
    assert res["fact_version"] == FACT_VERSION

    # Illegal tool calls
    with pytest.raises(IllegalToolNameError):
        tool.execute_tool("shell_exec", {"command": "dir"})

    with pytest.raises(IllegalToolNameError):
        tool.execute_tool("get_all_facts", {})

    with pytest.raises(IllegalToolNameError):
        tool.execute_tool("eval", {"code": "1+1"})
