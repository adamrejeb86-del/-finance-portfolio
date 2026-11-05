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
| # | Project | Highlights | Run |
|---|---|---|---|
| 01 | [Vanilla option pricing](01-options-pricing) | Black-Scholes, analytical Greeks, implied vol solver, American options on a binomial tree | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/01-options-pricing/options_pricing.ipynb) |
| 02 | [Phoenix autocall pricer](02-autocall-pricer) | Worst-of Monte Carlo, memory coupons, Greeks with common random numbers, fair-coupon grid | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/02-autocall-pricer/autocall_pricer.ipynb) |
| 03 | [Volatility surface](03-volatility-surface) | Heston Fourier pricing, SVI smiles with arbitrage checks, calibration to a full surface | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/03-volatility-surface/volatility_surface.ipynb) |
| 04 | [Exotic options & variance reduction](04-exotic-options-monte-carlo) | Asian, barrier and lookback options; control variates and Sobol cut the error 30 to 50 times | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/04-exotic-options-monte-carlo/exotic_options.ipynb) |
| 05 | [Structured notes](05-structured-notes) | Capital-protected notes and reverse convertibles built from a bond plus options | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/05-structured-notes/structured_notes.ipynb) |
| 06 | [Delta hedging](06-delta-hedging) | Discrete hedging error, realised vs implied vol, gamma-theta P&L attribution, transaction costs | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/06-delta-hedging/delta_hedging.ipynb) |
| 07 | [Convertible bonds](07-convertible-bond) | Tsiveriotis-Fernandes tree with equity and credit parts, calls, puts, implied credit spread | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/07-convertible-bond/convertible_bond.ipynb) |

### Fixed income and rates
| # | Project | Highlights | Run |
|---|---|---|---|
| 08 | [Yield curve & swaps](08-yield-curve-swaps) | Bootstrapping, Nelson-Siegel-Svensson, swap DV01 and key-rate risk, bond duration and convexity | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/08-yield-curve-swaps/yield_curve_swaps.ipynb) |
| 09 | [Interest rate models](09-interest-rate-models) | Hull-White fitted to the curve, caps and Jamshidian swaptions, calibration to normal vols | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/09-interest-rate-models/interest_rate_models.ipynb) |
| 10 | [ALM & immunisation](10-bond-portfolio-immunization) | Duration, Redington, key-rate and cash-flow matching of a pension liability under curve stresses | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/10-bond-portfolio-immunization/bond_portfolio_immunization.ipynb) |

### Credit and risk management
| # | Project | Highlights | Run |
|---|---|---|---|
| 11 | [CDS & Merton model](11-credit-risk-cds-merton) | Hazard-rate bootstrapping, standard-coupon CDS upfront and CS01, structural model, credit spread puzzle | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/11-credit-risk-cds-merton/credit_risk_cds_merton.ipynb) |
| 12 | [Credit portfolio & CDO tranches](12-credit-portfolio-copula) | Gaussian and t copulas, credit VaR vs Basel IRB capital, concentration risk, tranche spreads vs correlation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/12-credit-portfolio-copula/credit_portfolio_copula.ipynb) |
| 13 | [Credit scoring: scorecard vs ML](13-credit-scoring-ml) | WoE/IV, points scorecard, gradient boosting, Gini, calibration, master scale, PSI | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/13-credit-scoring-ml/credit_scoring.ipynb) |
| 14 | [VaR backtesting & stress tests](14-market-risk-var-backtesting) | Four VaR models, Kupiec, Christoffersen, Basel traffic light, ES backtest, reverse stress test | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/14-market-risk-var-backtesting/market_risk_var_backtesting.ipynb) |
| 15 | [Counterparty risk & CVA](15-counterparty-risk-cva) | Hull-White exposure simulation, EE/PFE, netting, collateral, CVA/DVA, wrong-way risk | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/15-counterparty-risk-cva/counterparty_risk_cva.ipynb) |
| 16 | [GARCH volatility forecasting](16-garch-volatility-forecasting) | GJR-GARCH-t by maximum likelihood, out-of-sample contest against EWMA, leverage effect, VaR | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/16-garch-volatility-forecasting/garch_volatility.ipynb) |

### Asset management
| # | Project | Highlights | Run |
|---|---|---|---|
| 17 | [Risk parity & Black-Litterman](17-risk-parity-black-litterman) | Markowitz instability, equal risk contribution, views blended with equilibrium | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/17-risk-parity-black-litterman/risk_parity_black_litterman.ipynb) |
| 18 | [Factor models & attribution](18-factor-models-attribution) | Alpha vs factor exposure with Newey-West errors, style drift, Brinson-Fachler with Carino linking | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamrejeb86-del/-finance-portfolio/blob/main/18-factor-models-attribution/factor_models_attribution.ipynb) |

## How to view and run
- **Read:** open any project folder, then its `.ipynb` notebook. GitHub shows the code, explanations and charts with all outputs.
- **Run live:** click **Open in Colab**, then *Runtime > Run all*. The first cell downloads the project code, so nothing needs installing.
- **If GitHub does not render a notebook:** use the nbviewer link in the project's README.

## Run locally
```bash
git clone https://github.com/adamrejeb86-del/-finance-portfolio.git
cd -finance-portfolio
pip install -r requirements.txt
cd 01-options-pricing && pytest -q && jupyter notebook
```
Each notebook also has a first cell that makes it runnable on Google Colab.
