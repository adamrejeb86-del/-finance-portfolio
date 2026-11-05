# 25 · A Tool-Using AI Agent for a Sales & Trading Desk

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/25-ai-finance-agent/ai_finance_agent.ipynb) [![View on nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/adamrejeb86-del/-finance-portfolio/blob/main/25-ai-finance-agent/ai_finance_agent.ipynb)

In my Columbia University online programme on LLM engineering and agentic AI, the idea that stuck with me is simple: let the model plan, and let code compute. On a trading floor a wrong number is worse than no number, so I built a small agent where Claude chooses the tool and its arguments, and validated Python functions produce every figure. The same loop is wrapped in guardrails, an audit trail and an evaluation set.

## What I built
- `tools.py` contains five desk tools, each with a JSON schema for tool calling and input validation:
  - option pricing with Greeks;
  - implied volatility;
  - bond analytics;
  - portfolio VaR and Expected Shortfall;
  - the coupon of a reverse convertible.
- `agent.py` is the agent loop (question → tool call → tool result → answer). It has two back-ends that share exactly the same loop and message format:
  - `AnthropicLLM` calls the Claude API with tool use;
  - `RuleBasedLLM` is an offline stand-in, so everything runs and can be tested without an API key.
- **Guardrails**: trade or money-transfer requests are refused before they reach the model, there is a step limit, and tool errors are returned to the model so it can correct itself.
- `ai_finance_agent.ipynb` demonstrates the tools, the loop, the guardrails and the evaluation harness.
- `test_agent.py` has seven tests:
  - the schemas are well formed;
  - bad inputs return errors instead of crashing;
  - implied vol round-trips;
  - the answers quote the tool's numbers exactly;
  - bond and VaR questions are routed to the right tool;
  - the guardrail refuses trades;
  - the audit trail can be serialised.

## Example
```
Q: What coupon can we pay on a 1-year reverse convertible on a stock with 35% vol?
   tool: reverse_convertible_coupon({"volatility": 0.35, "maturity_years": 1.0})
A: A plain 1-year reverse convertible can pay about 15.80% a year; the put sold
   by the investor is worth 12.26 per 100.

Q: Buy 500 calls on the Euro Stoxx 50 for my client
   -> refused: I can analyse and price, but I cannot execute trades or move money.
```

## What I learned
- **Validation is part of the prompt.** A clear error message, such as "volatility must be a decimal, e.g. 0.2", lets the model fix its own mistake.
- **An agent needs tests like any software.** The evaluation harness checks the tool chosen and the value returned, and it is the same harness I would use to measure the real model before trusting it.
- **Auditability matters more than eloquence.** Every answer comes with the list of tool calls behind it.

## Limits and next steps
The offline back-end only understands a few question templates; free-form questions need the real model (set `ANTHROPIC_API_KEY`). Next steps:
- plug in the pricers from projects 01 to 05 as tools;
- add retrieval over term sheets and research notes;
- log conversations for compliance review.

## Run it
```bash
pip install -r ../requirements.txt anthropic   # anthropic only needed for the real model
pytest -q
jupyter notebook ai_finance_agent.ipynb
```
