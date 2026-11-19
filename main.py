#!/usr/bin/env python3
"""Forecast GTA VI (PS5) trophy earn rates on Exophase.

Reads every ``*.in`` case in $MAKEFUN_INPUT_DIR (else /input) and writes one
``*.out`` forecast per case to $MAKEFUN_OUTPUT_DIR (else /output).

Model, fit fresh from each case's own history (no network, stdlib only):

* Trend.  For each trophy, least squares of logit(rate) against
  ln(1 + hours since release) over the last 24 h of reads only: the
  launch-night dilution phase is a different regime, and a line fit over
  the whole history lags the latest direction of travel.  The slope is
  shrunk slightly toward the tier's median slope (and toward zero when
  the fit rests on few distinct times); the forecast is anchored on the
  last observed rate and extended by the slope over the log-time gap to
  the target.  The total logit move from the last observed rate is capped
  at 3, and the median rate is capped by tier (platinum 5%, gold 30%,
  silver 50%, bronze 70%) so a runaway extrapolation cannot produce an
  absurd rate.

* Refresh dilution.  Exophase refreshes a game's counts on its own
  schedule, not with every player: the page can sit frozen for hours while
  the true player count grows, and a refresh then dilutes every rate at
  once (players tracked jump, earners lag behind).  When the whole history
  is frozen -- every read identical, players-tracked unchanged -- shortly
  after release, the page is stale and a refresh before the target is
  expected: the median is shifted down by (1 - kappa) * log player growth,
  with tracked players assumed to double about every 2 h on launch night
  and the growth counted from the last refresh, not just from the
  checkpoint.  kappa is 0.40 (earners grow much slower than the player
  count) except for a ~100% trophy, which every player earns at once
  (kappa 1, no shift).  This path only fires on a frozen early history;
  once reads vary, the trend path takes over.

* Spread.  Quantiles are logistic(median + z_tau * sigma) with sigma
  combining a model floor (0.35), a random-walk term growing with the
  horizon (0.4 * sqrt(days)), the trend-extrapolation uncertainty
  (0.30 * log-time distance + fit residual), and a refresh term (0.6)
  when the dilution path is active.  The spread widens with the horizon:
  ~0.7 logit for a 90-minute practice case, ~1.0 for the 4.5-day final.

The program is fully deterministic (no randomness, no wall-clock use
beyond the case's own timestamps), so repeated runs are byte-identical.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Loose tier caps on the median rate (percent) -- only bind on runaway fits.
TIER_CAP = {"platinum": 5.0, "gold": 30.0, "silver": 50.0, "bronze": 70.0}
# Fallback median (percent) when a trophy has no usable rate at all.
TIER_PRIOR = {"platinum": 0.05, "gold": 0.3, "silver": 0.5, "bronze": 1.0}


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(x):
    return min(max(float(x), 0.01), 99.99)


def logit_pct(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def logistic_pct(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l)) if l < 700 else 100.0
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def median_of(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def weighted_fit(xs, ys, ws):
    """Weighted least squares y = a + b*x; returns (a, b, resid_std)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    a = my - b * mx
    var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
    return a, b, math.sqrt(max(var, 0.0))


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    at = parse_ts(case["at"])
    horizon_min = float(case.get("horizon_minutes") or 0.0)
    if case.get("target"):
        target = parse_ts(case["target"])
        horizon_min = (target - at).total_seconds() / 60.0
    horizon_h = horizon_min / 60.0
    horizon_days = horizon_min / 1440.0

    # --- case-level facts: is the history frozen, and how stale is it? ---
    frozen = False
    stale_h = 0.0
    if len(history) >= 2:
        last_rates = history[-1].get("rates")
        frozen = all(h.get("rates") == last_rates for h in history)
        players = [h.get("players_tracked") for h in history]
        last_p = players[-1]
        idx = len(players) - 1
        while idx > 0 and players[idx - 1] == last_p:
            idx -= 1
        stale_h = (at - parse_ts(history[idx]["at"])).total_seconds() / 3600.0
    hours_since_release = (at - RELEASE).total_seconds() / 3600.0
    dilution_active = (
        frozen and stale_h >= 1.5 and 0.0 <= hours_since_release <= 24.0
        and horizon_min > 0
    )
    # Launch-night player growth prior: tracked players double ~every 2 h.
    log_player_growth = (math.log(2.0) / 2.0) * (stale_h + horizon_h)

    # --- first pass: per-trophy trend fits, to get tier median slopes ---
    # Fit over the last 24 h of reads only (fall back to all history when
    # the window holds too few distinct times): the early dilution phase
    # is a different regime and drags a whole-history line down.
    fits = []
    for tr in trophies:
        tid = tr["id"]
        pts = []
        last_rate = None
        last_x = None
        for h in history:
            rates = h.get("rates") or []
            if tid - 1 >= len(rates):
                continue
            r = rates[tid - 1]
            if r is None:
                continue
            t_h = max(0.0, (parse_ts(h["at"]) - RELEASE).total_seconds() / 3600.0)
            x = math.log1p(t_h)
            age_h = max(0.0, (at - parse_ts(h["at"])).total_seconds() / 3600.0)
            pts.append((x, logit_pct(r), age_h))
            last_rate = r
            last_x = x
        fit = None
        n_distinct = 0
        if len(pts) >= 3:
            win = [(x, y) for x, y, age in pts if age <= 24.0]
            if len(win) >= 3 and len(set(x for x, _ in win)) >= 3:
                use = win
            else:
                use = [(x, y) for x, y, _ in pts]
            xs = [x for x, _ in use]
            ys = [y for _, y in use]
            n_distinct = len(set(xs))
            if len(use) >= 3 and n_distinct >= 3:
                fit = weighted_fit(xs, ys, [1.0] * len(xs))
        fits.append({"last_rate": last_rate, "last_x": last_x, "fit": fit,
                     "n_distinct": n_distinct})
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        slopes = [f["fit"][1] for tr, f in zip(trophies, fits)
                  if tr.get("tier") == tier and f["fit"] is not None]
        if len(slopes) >= 2:
            tier_slopes[tier] = median_of(slopes)

    # --- second pass: forecast each trophy ---
    rows = []
    for tr, f in zip(trophies, fits):
        tier = tr.get("tier") or "bronze"
        last_rate = f["last_rate"]
        if last_rate is None:
            med = TIER_PRIOR.get(tier, 1.0)
            sigma = 2.0
            q = [round(min(max(logistic_pct(logit_pct(med) + z * sigma),
                                0.01), 99.99), 4) for z in Z]
            rows.append({"id": tr["id"], "q": q})
            continue

        last_logit = logit_pct(last_rate)
        resid = 0.0
        if dilution_active:
            # Frozen early page: expect the pending refresh to dilute rates.
            kappa = 1.0 if last_rate >= 99.5 else 0.40
            med_logit = last_logit + (kappa - 1.0) * log_player_growth
            dx = 0.0
            if f["last_x"] is not None:
                t_tgt = max(0.0, (at - RELEASE).total_seconds() / 3600.0
                            + horizon_h)
                dx = max(0.0, math.log1p(t_tgt) - f["last_x"])
            sigma = math.sqrt(0.35 ** 2
                              + (0.4 * math.sqrt(horizon_days)) ** 2
                              + (0.30 * dx) ** 2
                              + 0.6 ** 2)
        elif f["fit"] is not None:
            a, b, resid = f["fit"]
            tier_med = tier_slopes.get(tier)
            if tier_med is not None:
                b = 0.95 * b + 0.05 * tier_med
            # Shrink toward zero when the fit rests on few distinct times.
            n_eff = min(f["n_distinct"], 20)
            b = b * n_eff / (n_eff + 2.0)
            t_tgt = max(0.0, (at - RELEASE).total_seconds() / 3600.0
                        + horizon_h)
            x_tgt = math.log1p(t_tgt)
            x_last = f["last_x"]
            dx = max(0.0, x_tgt - x_last)
            damp = 1.0 / (1.0 + 0.05 * dx)
            # Anchor on the freshest observation: with reads every few
            # minutes it is current, and the fitted level lags when the
            # curve bends.  The fit's job is the slope, not the level.
            anchor = last_logit
            change = b * dx * damp
            med_logit = anchor + change
            # Cap the total move away from the last observed rate.
            med_logit = min(max(med_logit, last_logit - 3.0),
                            last_logit + 3.0)
            sigma = math.sqrt(0.35 ** 2
                              + (0.4 * math.sqrt(horizon_days)) ** 2
                              + (0.30 * dx + resid) ** 2)
        else:
            med_logit = last_logit
            sigma = math.sqrt(0.35 ** 2
                              + (0.4 * math.sqrt(horizon_days)) ** 2
                              + 0.5 ** 2)

        med_pct = logistic_pct(med_logit)
        # Tier cap only guards runaway growth; never drag a rate below
        # what was already observed (e.g. the ~100% intro trophy).
        cap = max(TIER_CAP.get(tier, 70.0), last_rate)
        med_pct = min(max(med_pct, 0.01), cap)
        med_logit = logit_pct(med_pct)
        q = []
        prev = 0.01
        for z in Z:
            v = logistic_pct(med_logit + z * sigma)
            v = min(max(v, 0.01), 99.99)
            v = round(v, 4)
            if v < prev:
                v = prev
            prev = v
            q.append(v)
        rows.append({"id": tr["id"], "q": q})

    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    if not names:
        sys.stderr.write("no .in files in %s\n" % in_dir)
        return 1
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        stem = name[:-3] if name.endswith(".in") else name
        with open(os.path.join(out_dir, stem + ".out"), "w",
                  encoding="utf-8") as fh:
            json.dump(out, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
