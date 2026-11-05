"""
A tool-using finance agent: the language model plans, the Python tools compute.

Loop (the standard agentic pattern):
  1. Send the user's question, the system prompt and the tool definitions to the model.
  2. If the model asks for tools, validate and run them, append the results, go back to 1.
  3. When the model answers in text, return it with the full audit trail of tool calls.

Guardrails
  - Requests to execute trades or move money are refused before reaching the model.
  - A maximum number of steps stops runaway loops.
  - Tool errors (bad inputs) are returned to the model so it can correct itself.

Two back-ends share exactly the same loop:
  - `AnthropicLLM`: the Claude API (needs the `anthropic` package and ANTHROPIC_API_KEY);
  - `RuleBasedLLM`: an offline stand-in that parses a few question templates, so the whole
    pipeline runs and is testable without an API key. It is clearly not intelligent; it
    exists to exercise the agent loop deterministically.

Author: Adam Rejeb
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from tools import call_tool, tool_specs

SYSTEM_PROMPT = """You are a quantitative finance assistant for a sales and trading desk.
Always use the tools for any number: never compute prices, risks or coupons yourself.
Rates and volatilities are decimals (20% = 0.2). State the assumptions you used.
You cannot execute trades, send orders or move money; say so if asked."""

BLOCKED = re.compile(r"\b(buy|sell|execute|place an order|send (an )?order|transfer|wire)\b", re.I)


@dataclass
class ToolCall:
    name: str
    arguments: dict
    result: dict


@dataclass
class AgentResult:
    answer: str
    calls: list = field(default_factory=list)
    steps: int = 0
    refused: bool = False


# ---------------------------------------------------------------- back-ends
class AnthropicLLM:
    """Thin wrapper around the Anthropic Messages API with tool use."""

    def __init__(self, model: str | None = None, max_tokens: int = 1024):
        import anthropic                                               # imported only when the API is used
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def respond(self, messages: list) -> dict:
        msg = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=SYSTEM_PROMPT,
                                          tools=tool_specs(), messages=messages)
        return {"stop_reason": msg.stop_reason, "content": [b.model_dump() for b in msg.content]}


class RuleBasedLLM:
    """Offline stand-in with the same interface. It recognises a handful of templates, asks for
    the matching tool, then phrases the tool result. Useful for tests and demos without a key."""

    def respond(self, messages: list) -> dict:
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], list) and last["content"][0].get("type") == "tool_result":
            result = json.loads(last["content"][0]["content"])
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": self._phrase(messages, result)}]}
        q = messages[-1]["content"] if isinstance(messages[-1]["content"], str) else ""
        call = self._plan(q)
        if call is None:
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text":
                    "I can price options, invert implied volatility, analyse bonds, compute a portfolio VaR "
                    "and quote a reverse convertible coupon. Could you rephrase with the inputs?"}]}
        return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "t1", "name": call[0], "input": call[1]}]}

    @staticmethod
    def _num(pattern, text, default=None, pct=False):
        m = re.search(pattern, text, re.I)
        if not m:
            return default
        v = float(m.group(1).replace(",", ""))
        return v / 100 if pct else v

    def _maturity(self, q, default=1.0):
        months = self._num(r"([\d.]+)\s*-?\s*month", q)
        return months / 12 if months is not None else self._num(r"([\d.]+)\s*-?\s*year", q, default)

    def _plan(self, q):
        n = self._num
        if re.search(r"implied vol", q, re.I):
            return "implied_volatility", {"option_price": n(r"(?:price|worth|trades at)\s*(?:of\s*)?([\d.]+)", q), "spot": n(r"spot\s*(?:of\s*|at\s*)?([\d.]+)", q),
                                          "strike": n(r"strike\s*(?:of\s*)?([\d.]+)", q), "maturity_years": self._maturity(q),
                                          "option_type": "put" if re.search(r"\bput\b", q, re.I) else "call"}
        if re.search(r"\b(call|put)\b", q, re.I) and re.search(r"strike", q, re.I):
            return "price_option", {"spot": n(r"spot\s*(?:of\s*|at\s*)?([\d.]+)", q), "strike": n(r"strike\s*(?:of\s*)?([\d.]+)", q),
                                    "maturity_years": self._maturity(q), "volatility": n(r"([\d.]+)\s*%\s*vol", q, pct=True),
                                    "rate": n(r"rates?\s*(?:of\s*|at\s*)?([\d.]+)\s*%", q, 3.0, pct=True) if re.search(r"rate", q, re.I) else 0.03,
                                    "option_type": "put" if re.search(r"\bput\b", q, re.I) else "call"}
        if re.search(r"\bvar\b|value at risk", q, re.I):
            weights = [float(x) / 100 for x in re.findall(r"([\d.]+)\s*%\s*in", q)]
            vols = [float(x) / 100 for x in re.findall(r"([\d.]+)\s*%\s*vol", q)]
            return "portfolio_var", {"weights": weights, "volatilities": vols, "correlation": n(r"correlation\s*(?:of\s*)?([-\d.]+)", q, 0.3),
                                     "portfolio_value": n(r"(?:EUR|€)\s*([\d,.]+)\s*m", q, 1.0) * 1e6,
                                     "confidence": n(r"([\d.]+)\s*%\s*(?:confidence|VaR)", q, 99, pct=True)}
        if re.search(r"bond", q, re.I):
            return "bond_analytics", {"coupon_rate": n(r"([\d.]+)\s*%\s*coupon", q, pct=True), "maturity_years": int(n(r"([\d.]+)\s*-?\s*year", q, 10)),
                                      "yield_to_maturity": n(r"yield\s*(?:of\s*)?([\d.]+)\s*%", q, pct=True)}
        if re.search(r"reverse convertible", q, re.I):
            return "reverse_convertible_coupon", {"volatility": n(r"([\d.]+)\s*%\s*vol", q, pct=True), "maturity_years": self._maturity(q)}
        return None

    @staticmethod
    def _phrase(messages, result):
        if "error" in result:
            return f"I could not compute this: {result['error']}."
        name = messages[-2]["content"][0]["name"]
        r = result
        if name == "price_option":
            return f"The option is worth {r['price']:.2f}, with a delta of {r['delta']:.2f} and a vega of {r['vega_per_vol_point']:.2f} per vol point (Black-Scholes)."
        if name == "implied_volatility":
            return f"The implied volatility is {r['implied_volatility']:.2%}."
        if name == "bond_analytics":
            return f"The bond prices at {r['price']:.2f}, with a modified duration of {r['modified_duration']:.2f} and a DV01 of {r['dv01_per_100']:.4f} per 100 nominal."
        if name == "portfolio_var":
            return (f"The portfolio's annual volatility is {r['portfolio_volatility_annual']:.2%}; the 1-day VaR is EUR {r['value_at_risk']:,.0f} "
                    f"and the Expected Shortfall EUR {r['expected_shortfall']:,.0f}.")
        if name == "reverse_convertible_coupon":
            return f"A plain 1-year reverse convertible can pay about {r['coupon_per_year']:.2%} a year; the put sold by the investor is worth {r['put_value_per_100']:.2f} per 100."
        return json.dumps(r)


# ---------------------------------------------------------------- the agent loop
def run_agent(question: str, llm=None, max_steps: int = 6) -> AgentResult:
    if BLOCKED.search(question):
        return AgentResult("I can analyse and price, but I cannot execute trades or move money.", refused=True)
    llm = llm or RuleBasedLLM()
    messages = [{"role": "user", "content": question}]
    calls = []
    for step in range(1, max_steps + 1):
        reply = llm.respond(messages)
        messages.append({"role": "assistant", "content": reply["content"]})
        if reply["stop_reason"] != "tool_use":
            text = " ".join(b["text"] for b in reply["content"] if b["type"] == "text")
            return AgentResult(text, calls, step)
        results = []
        for block in reply["content"]:
            if block["type"] != "tool_use":
                continue
            out = call_tool(block["name"], block["input"])
            calls.append(ToolCall(block["name"], block["input"], out))
            results.append({"type": "tool_result", "tool_use_id": block["id"], "content": json.dumps(out)})
        messages.append({"role": "user", "content": results})
    return AgentResult("Stopped: too many steps.", calls, max_steps)


def evaluate(cases: list, llm=None) -> list:
    """Run a list of {'question', 'tool', 'check'} cases; `check` is a function of the tool result."""
    rows = []
    for c in cases:
        res = run_agent(c["question"], llm)
        used = res.calls[0].name if res.calls else None
        ok_tool = used == c["tool"]
        ok_value = bool(res.calls) and "error" not in res.calls[0].result and c["check"](res.calls[0].result)
        rows.append({"question": c["question"], "expected tool": c["tool"], "tool used": used,
                     "correct tool": ok_tool, "correct value": ok_value, "answer": res.answer})
    return rows
