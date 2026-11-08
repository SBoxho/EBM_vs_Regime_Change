# Do glass-box explanations survive regime change?

When an Explainable Boosting Machine (EBM) is retrained over time on intraday crypto market data, do its
explanations stay stable, or do they change with the market regime?

Glass-box models are often presented as showing "what the model learned about the market". This project tests
that claim:

- **Retraining.** EBMs are retrained on rolling 12-month windows of hourly Binance spot data (BTC, ETH and XRP from
  2018, SOL from 2020, all through 2026), forecasting 8 hours ahead, with the feature set and hyperparameters held
  fixed.
- **Measurement.** We measure how much the shape functions, term importances and selected interactions move
  between windows. Every change is compared with a measured noise floor: the spread between retrains on the same
  window under different seeds and resamples.
- **Controls.** A learnable target (forward realised volatility) sits next to the near-noise target (forward
  return direction), and a shuffled target shows what pure-noise drift looks like. Together they separate "the
  market changed" from "there was never a signal".
- **Secondary questions.** Is drift concentrated around volatility, trend and liquidity regime changes? Does it
  precede out-of-sample decay? Is it specific to EBM, compared with logistic regression and XGBoost + SHAP?

**Status:** M0 (environment and data) done. M1 (features, targets, regimes, splits) is next. No models have been fitted yet.

## Reproduce the data audit

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock      # on Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pip install --no-deps -e .
.venv/Scripts/python scripts/download_data.py                  # about 74 MB, checksum-verified
.venv/Scripts/python scripts/build_bars.py
.venv/Scripts/python -m pytest                                 # unit tests plus data acceptance checks
```

Then open `notebooks/00_data_audit.ipynb`.

## Documents

- [`docs/PLAN.md`](docs/PLAN.md): research and implementation plan, including pre-registered hypotheses.
- [`docs/DECISIONS.md`](docs/DECISIONS.md): design decisions and their reasons.
- [`docs/LITERATURE.md`](docs/LITERATURE.md): verified references.
- [`scripts/time_fits.py`](scripts/time_fits.py): the timing script behind the compute estimate (synthetic data only).

**Data notice.** Market data comes from the [Binance public data archive](https://data.binance.vision). It is
licensed under CC BY-NC-SA 4.0 by the Binance Vision Dataset Terms, which do not allow hosting it. This repository
will therefore ship download-and-verify scripts only, never the data. It is not affiliated with or endorsed by
Binance.
