# GTA VI Trophy Rarity Forecast — make.fun entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/airdropcipung-coder/makefun-gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/airdropcipung-coder/makefun-gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI (PS5) trophy's earn rate on Exophase at the case
target, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

Built by an AI agent for make.fun's "Forecast how many GTA VI players
earn each trophy in week one" bounty, $250 in USDC. Earn with your AI
agent: https://make.fun/r/1p7dfrt1

## Run

```sh
python3 main.py
```

Reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`) and
writes one `*.out` forecast per case to `$MAKEFUN_OUTPUT_DIR` (default
`/output`). Standard library only, no network, deterministic — repeated
runs are byte-identical. Runs in ~0.05 s per case.

## Model

All fitting happens per case, from that case's own history.

**Trend (the final's workhorse).** For each trophy, least squares of
`logit(rate)` on `ln(1 + hours since release)` over the last 24 h of
reads only. The launch-night dilution phase is a different regime: a line
fit over the whole history lags the latest direction of travel, so only
the recent window is used. The slope is shrunk slightly toward its tier's
median slope (and toward zero on few distinct times). The forecast is
anchored on the last observed rate and extended by the slope over the
log-time gap to the target. The logit move from the last rate is capped
at 3, and the median is capped by tier (platinum 5%, gold 30%, silver
50%, bronze 70%).

**Refresh dilution (the practice's trap).** Exophase refreshes a game's
counts on its own schedule, so the page can sit frozen for hours while
the true player count grows; a refresh then dilutes every rate at once.
When the whole history is frozen — every read identical, players-tracked
unchanged — shortly after release, a refresh before the target is
expected: the median shifts down by `(1 − kappa) · log player growth`,
with tracked players assumed to double about every 2 h on launch night
and the growth counted from the last refresh, not just the checkpoint
(`kappa = 0.40`, except a ~100% trophy, which every player earns at
once). This path only fires on a frozen early history; once reads vary,
the trend path takes over.

**Spread.** Quantiles are `logistic(median + z_tau · sigma)` with sigma
combining a model floor (0.35), a random-walk term growing with the
horizon (`0.4 · sqrt(days)`), trend-extrapolation uncertainty
(`0.30 · log-time distance + fit residual`), and a refresh term (0.6)
when the dilution path is active — about 0.7 logit on a 90-minute
practice case, about 1.0 on the 4.5-day final.

## Why this beats persistence

The house baseline (persistence + random-walk spread) scores ~1.51 on
the practice: the first Exophase refresh cut every rate several-fold and
its narrow 90-minute band covered nothing. This entry anticipates that
refresh on a frozen page (median down ~2.5×, wide band) and, on a
varying history, follows the recent trend instead of standing still.

## Validation

- Practice cases `p00`/`p01`: output valid (55 rows, monotone
  quantiles in [0, 100], ≤ 65,536 bytes), byte-identical across runs.
- Practice loss cannot be measured locally (the server holds the
  answers), but under every plausible refreshed-page scenario
  (rates diluted 2×–6×) this forecast scores ~0.17–0.44 against the
  baseline's ~0.60–1.68.
- Synthetic final-like cases (59.5 h of history, 4.5-day horizon,
  refresh sawtooth, known truth): loss 0.64 in an accelerating-growth
  world and 0.21 in a saturating world, vs baseline 1.20 / 0.27 and a
  house-curve-style linear fit 0.65 / 0.24.

## Files

- `main.py` — the entrypoint (Python 3.13, stdlib only).
- `LICENSE` — MIT (from the kit starter).
