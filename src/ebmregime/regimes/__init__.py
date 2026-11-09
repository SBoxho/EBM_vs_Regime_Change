"""Market regimes. ``causal``: R1/R2/R3 computed daily and applied to the next day's hours (D-20).

The primary R1 column is pre-registered in ``configs/prereg.yaml`` (``regimes.primary``, D-44:
``r1_vol_2y``); read it with :func:`primary_r1` rather than hard-coding it.
"""

from ebmregime.regimes.causal import (daily_inputs, daily_regimes, percentile_rank, primary_r1,
                                      regime_change_covariates, to_hourly, trend_tstat,
                                      window_shares)

__all__ = ["daily_inputs", "daily_regimes", "percentile_rank", "primary_r1",
           "regime_change_covariates", "to_hourly", "trend_tstat", "window_shares"]
