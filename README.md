# Do glass-box explanations survive regime change?

When an Explainable Boosting Machine (EBM) is retrained over time on intraday crypto market data, do its
explanations stay stable, or do they change with the market regime?

**Status:** planning. The plan will be in [`docs/PLAN.md`](docs/PLAN.md).

---

## TODO

### Future project: who forecasts football best, prediction markets, bookmakers or Elo?

Compare, on the same football matches, three sources of match probabilities:

- **Prediction markets:** Polymarket (public API), and Kalshi if football markets are available.
- **Bookmakers:** opening and closing odds (for example football-data.co.uk), with the margin removed.
- **A plain Elo model:** built the same way as in `glass-box_World_Cup_predictor`.

Questions:

- Which source is best calibrated? Use reliability diagrams, Brier score and log-loss, with bootstrap
  confidence intervals.
- Where does each source fail? Look at the favourite–longshot bias, thin or low-volume markets, draws,
  and smaller leagues against the top leagues.
- How do prices move as kick-off approaches? Is the prediction-market price ahead of or behind the
  bookmaker line?
- Does volume or liquidity explain when prediction markets are well or badly calibrated?
- Does a simple combination (for example, logit averaging of market and Elo) beat each source on its
  own, out of sample?

Before starting:

- Check how much historical football data the Polymarket API provides (markets, price history, volume).
- Find a reliable way to match matches across the three sources (team names, kick-off times).
- Decide which leagues and seasons have enough markets for honest confidence intervals.
- Keep it to public data, with no betting or trading, and treat it as a calibration study only.
