# Quantitative Finance Portfolio · Adam Rejeb

Python projects covering derivatives pricing, structured products, fixed income, credit, risk management, asset management, corporate finance and automation.

Master's in Finance from Université Paris II Panthéon-Assas (ranked 1st in class) · 2 years at Amundi (portfolio management, structured products) · Sales at Montpensier Finance · AMF certified.

## How these projects are built
- **Explained, not just coded.** Each notebook walks through the financial intuition, the maths and the results, with charts.
- **Verified.** Every model is tested against closed-form solutions, limiting cases or an independent method. Run `pytest` in any folder.
- **Executed.** Notebooks are saved with their outputs, so the results can be read on GitHub without running anything.
- **Synthetic data.** Market data is simulated or illustrative, and no data from any employer is used.

## Projects

### Derivatives pricing and structured products
| # | Project | Highlights |
|---|---|---|
| 01 | [Vanilla option pricing](01-options-pricing) | Black-Scholes, analytical Greeks, implied vol solver, American options on a binomial tree |
| 02 | [Phoenix autocall pricer](02-autocall-pricer) | Worst-of Monte Carlo, memory coupons, Greeks with common random numbers, fair-coupon grid |
| 03 | [Volatility surface](03-volatility-surface) | Heston Fourier pricing, SVI smiles with arbitrage checks, calibration to a full surface |
| 04 | [Exotic options & variance reduction](04-exotic-options-monte-carlo) | Asian, barrier and lookback options; control variates and Sobol cut the error 30 to 50 times |
| 05 | [Structured notes](05-structured-notes) | Capital-protected notes and reverse convertibles built from a bond plus options |

## Run locally
```bash
git clone https://github.com/adamrejeb86-del/-finance-portfolio.git
cd -finance-portfolio
pip install -r requirements.txt
cd 01-options-pricing && pytest -q && jupyter notebook
```
Each notebook also has a first cell that makes it runnable on Google Colab.
