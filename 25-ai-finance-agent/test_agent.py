import json

import pytest

from agent import RuleBasedLLM, run_agent
from tools import bond_analytics, call_tool, implied_volatility, portfolio_var, price_option, tool_specs


def test_tool_specs_are_well_formed():
    for spec in tool_specs():
        assert {"name", "description", "input_schema"} <= set(spec)
        assert set(spec["input_schema"]["required"]) <= set(spec["input_schema"]["properties"])


def test_tool_validation_returns_errors_instead_of_crashing():
    assert "error" in call_tool("price_option", {"spot": 100, "strike": 100, "maturity_years": 1, "volatility": 20})
    assert "error" in call_tool("price_option", {"spot": 100})
    assert "error" in call_tool("unknown", {})


def test_implied_vol_round_trip():
    p = price_option(100, 110, 0.5, 0.27)["price"]
    assert implied_volatility(p, 100, 110, 0.5)["implied_volatility"] == pytest.approx(0.27, abs=1e-3)


def test_agent_answers_with_tool_numbers():
    res = run_agent("Price a 1-year call with spot 100, strike 105 and 25% vol")
    assert res.calls[0].name == "price_option"
    expected = price_option(100, 105, 1.0, 0.25)["price"]
    assert f"{expected:.2f}" in res.answer and res.steps == 2


def test_agent_handles_bond_and_var_questions():
    bond = run_agent("What is the price of a 10-year bond with a 3% coupon at a yield of 3.5%?")
    assert bond.calls[0].result == bond_analytics(0.03, 10, 0.035)
    var = run_agent("Compute the 99% VaR of EUR 10m with 60% in equities at 18% vol and 40% in bonds at 6% vol, correlation 0.2")
    assert var.calls[0].result == portfolio_var([0.6, 0.4], [0.18, 0.06], 0.2, 1e7, 0.99)


def test_guardrail_refuses_trading_requests():
    res = run_agent("Buy 100 calls on the CAC 40 for me")
    assert res.refused and not res.calls


def test_audit_trail_is_serialisable():
    res = run_agent("Price a 2-year put with spot 50, strike 50 and 30% vol")
    json.dumps([c.__dict__ for c in res.calls])
