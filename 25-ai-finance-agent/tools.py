"""
Finance tools exposed to the agent. Each tool is a plain Python function plus a JSON schema
describing its inputs, in the format used by LLM tool-calling APIs.

The tools are deliberately deterministic and validated: the language model decides *which*
tool to call and with which arguments, but every number in an answer comes from code.

Author: Adam Rejeb
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


class ToolError(ValueError):
    """Raised when arguments are missing or outside sensible bounds."""


def _check(cond, msg):
    if not cond:
        raise ToolError(msg)


def price_option(spot: float, strike: float, maturity_years: float, volatility: float, rate: float = 0.03,
                 dividend_yield: float = 0.0, option_type: str = "call") -> dict:
    _check(spot > 0 and strike > 0, "spot and strike must be positive")
    _check(0 < maturity_years <= 30, "maturity must be between 0 and 30 years")
    _check(0.01 <= volatility <= 3, "volatility must be between 1% and 300% (as a decimal, e.g. 0.2)")
    _check(option_type in ("call", "put"), "option_type must be 'call' or 'put'")
    T, s = maturity_years, volatility
    d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * s**2) * T) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)
    if option_type == "call":
        price = spot * np.exp(-dividend_yield * T) * norm.cdf(d1) - strike * np.exp(-rate * T) * norm.cdf(d2)
        delta = np.exp(-dividend_yield * T) * norm.cdf(d1)
    else:
        price = strike * np.exp(-rate * T) * norm.cdf(-d2) - spot * np.exp(-dividend_yield * T) * norm.cdf(-d1)
        delta = -np.exp(-dividend_yield * T) * norm.cdf(-d1)
    vega = spot * np.exp(-dividend_yield * T) * norm.pdf(d1) * np.sqrt(T) / 100
    return {"price": round(float(price), 4), "delta": round(float(delta), 4), "vega_per_vol_point": round(float(vega), 4)}


def implied_volatility(option_price: float, spot: float, strike: float, maturity_years: float, rate: float = 0.03,
                       option_type: str = "call") -> dict:
    _check(option_price > 0, "option price must be positive")
    f = lambda v: price_option(spot, strike, maturity_years, v, rate, 0.0, option_type)["price"] - option_price
    try:
        vol = brentq(f, 0.011, 2.99, xtol=1e-8)
    except ValueError:
        raise ToolError("no volatility between 1% and 300% matches this price (check arbitrage bounds)")
    return {"implied_volatility": round(float(vol), 4)}


def bond_analytics(coupon_rate: float, maturity_years: int, yield_to_maturity: float, face: float = 100.0) -> dict:
    _check(0 <= coupon_rate < 0.5 and -0.05 < yield_to_maturity < 0.5, "rates must be decimals, e.g. 0.03 for 3%")
    _check(1 <= int(maturity_years) <= 50, "maturity must be between 1 and 50 years")
    t = np.arange(1, int(maturity_years) + 1)
    cf = np.full(len(t), coupon_rate * face); cf[-1] += face
    pv = cf / (1 + yield_to_maturity) ** t
    price = pv.sum()
    mac = (t * pv).sum() / price
    conv = (t * (t + 1) * pv).sum() / (price * (1 + yield_to_maturity) ** 2)
    return {"price": round(float(price), 4), "modified_duration": round(float(mac / (1 + yield_to_maturity)), 4),
            "convexity": round(float(conv), 4), "dv01_per_100": round(float(mac / (1 + yield_to_maturity) * price * 1e-4), 5)}


def portfolio_var(weights: list, volatilities: list, correlation: float, portfolio_value: float,
                  confidence: float = 0.99, horizon_days: int = 1) -> dict:
    w, v = np.asarray(weights, float), np.asarray(volatilities, float)
    _check(len(w) == len(v) and len(w) >= 1, "weights and volatilities must have the same length")
    _check(abs(w.sum() - 1) < 1e-6, "weights must sum to 1")
    _check(-0.99 < correlation < 1, "correlation must be in (-1, 1)")
    _check(0.9 <= confidence < 1, "confidence must be between 0.90 and 0.999")
    corr = np.full((len(w), len(w)), correlation); np.fill_diagonal(corr, 1.0)
    cov = np.outer(v, v) * corr
    sigma_annual = float(np.sqrt(w @ cov @ w))
    sigma_h = sigma_annual * np.sqrt(horizon_days / 252)
    z = norm.ppf(confidence)
    return {"portfolio_volatility_annual": round(sigma_annual, 4),
            "value_at_risk": round(portfolio_value * z * sigma_h, 2),
            "expected_shortfall": round(portfolio_value * sigma_h * norm.pdf(z) / (1 - confidence), 2)}


def reverse_convertible_coupon(volatility: float, maturity_years: float = 1.0, rate: float = 0.03, issuer_spread: float = 0.01,
                               fees: float = 0.01) -> dict:
    """Annual coupon of a plain reverse convertible (bond + short at-the-money put)."""
    _check(0.05 <= volatility <= 1.5, "volatility must be a decimal between 0.05 and 1.5")
    put = price_option(100, 100, maturity_years, volatility, rate, 0.0, "put")["price"]
    df = np.exp(-(rate + issuer_spread) * np.arange(1, int(np.ceil(maturity_years)) + 1))
    coupon = (100 * (1 - fees) - 100 * df[-1] + put) / (100 * df.sum())
    return {"coupon_per_year": round(float(coupon), 4), "put_value_per_100": round(put, 4)}


TOOLS = {
    "price_option": (price_option, {
        "description": "Black-Scholes price, delta and vega of a European call or put. Volatility and rates as decimals.",
        "input_schema": {"type": "object", "properties": {
            "spot": {"type": "number"}, "strike": {"type": "number"}, "maturity_years": {"type": "number"},
            "volatility": {"type": "number"}, "rate": {"type": "number"}, "dividend_yield": {"type": "number"},
            "option_type": {"type": "string", "enum": ["call", "put"]}},
            "required": ["spot", "strike", "maturity_years", "volatility"]}}),
    "implied_volatility": (implied_volatility, {
        "description": "Implied volatility of a European option from its market price.",
        "input_schema": {"type": "object", "properties": {
            "option_price": {"type": "number"}, "spot": {"type": "number"}, "strike": {"type": "number"},
            "maturity_years": {"type": "number"}, "rate": {"type": "number"}, "option_type": {"type": "string", "enum": ["call", "put"]}},
            "required": ["option_price", "spot", "strike", "maturity_years"]}}),
    "bond_analytics": (bond_analytics, {
        "description": "Price, modified duration, convexity and DV01 of an annual-coupon bond from its yield.",
        "input_schema": {"type": "object", "properties": {
            "coupon_rate": {"type": "number"}, "maturity_years": {"type": "integer"}, "yield_to_maturity": {"type": "number"},
            "face": {"type": "number"}}, "required": ["coupon_rate", "maturity_years", "yield_to_maturity"]}}),
    "portfolio_var": (portfolio_var, {
        "description": "Parametric VaR and Expected Shortfall of a portfolio with a common correlation between assets.",
        "input_schema": {"type": "object", "properties": {
            "weights": {"type": "array", "items": {"type": "number"}}, "volatilities": {"type": "array", "items": {"type": "number"}},
            "correlation": {"type": "number"}, "portfolio_value": {"type": "number"}, "confidence": {"type": "number"},
            "horizon_days": {"type": "integer"}}, "required": ["weights", "volatilities", "correlation", "portfolio_value"]}}),
    "reverse_convertible_coupon": (reverse_convertible_coupon, {
        "description": "Annual coupon a bank can pay on a plain reverse convertible, given the underlying's volatility.",
        "input_schema": {"type": "object", "properties": {
            "volatility": {"type": "number"}, "maturity_years": {"type": "number"}, "rate": {"type": "number"},
            "issuer_spread": {"type": "number"}, "fees": {"type": "number"}}, "required": ["volatility"]}}),
}


def tool_specs() -> list:
    """Tool definitions in the format expected by the Anthropic Messages API."""
    return [{"name": name, **spec} for name, (_, spec) in TOOLS.items()]


def call_tool(name: str, arguments: dict) -> dict:
    """Validate and execute one tool call. Errors are returned to the model, not raised."""
    if name not in TOOLS:
        return {"error": f"unknown tool {name}"}
    fn, spec = TOOLS[name]
    arguments = {k: v for k, v in arguments.items() if v is not None}
    missing = [k for k in spec["input_schema"]["required"] if k not in arguments]
    if missing:
        return {"error": f"missing arguments: {missing}"}
    allowed = set(spec["input_schema"]["properties"])
    try:
        return fn(**{k: v for k, v in arguments.items() if k in allowed})
    except ToolError as e:
        return {"error": str(e)}
