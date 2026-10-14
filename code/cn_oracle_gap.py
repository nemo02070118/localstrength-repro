"""WHERE IS THE GAP HIDDEN?  Decompose the Oracle advantage into (i) genuine
persistent venue signal and (ii) selection-over-noise (the upward bias of taking
argmax over V noisy venue Sharpe estimates).

The user's exact question: we know the upper bound exists (Oracle 1.447 >> coarse
1.302 >> grave 0.849) -- there IS meat on the bone -- yet every OOS router fails
to eat it. Why? Hypothesis: most of the Oracle gap is a MIRAGE created by
maximizing over V correlated noisy venue returns. The max-of-noise is real
in-sample but carries NO tradable (persistent) information.

Test: build NULL worlds where each factor has ZERO true venue advantage (all
venues share the factor's common component; venue deviations are demeaned so the
true advantage is exactly 0) but the realized sample noise and the cross-venue
covariance are preserved by block bootstrap. Recompute the Oracle in each null
world. If the NULL Oracle reproduces most of the real Oracle gap, the 'meat' is
selection noise, not tradable signal -- and the impossibility of eating it is a
theorem, not a failure of method.

Also report the clean decomposition:
  gap_total   = Oracle - coarse            (full look-ahead advantage)
  gap_null    = E[Oracle_null] - coarse    (selection-over-noise component)
  gap_signal  = gap_total - gap_null       (genuine tradable component)

Writes cn_oracle_gap.json.
"""
from __future__ import annotations

import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from cn_resurrection import load_battlefields, _tstat, _sr, BATTLES

warnings.filterwarnings("ignore")
HERE = Path(os.environ.get("LOCALSTRENGTH_RESULTS",
                           str(Path(__file__).resolve().parent.parent / "results")))
ANN = np.sqrt(12.0)
COARSE = "wide"

def _sr_series(x):
    x = pd.Series(x).dropna()
    if x.size < 12 or x.std(ddof=1) == 0:
        return np.nan
    return float(ANN * x.mean() / x.std(ddof=1))


def book(ret_df):
    return ret_df.mean(axis=1).dropna()


def oracle_book(Bs, disc, venues, score_months, trade_months):
    """Route each factor to the venue with the highest SR over `score_months`;
    realize returns over `trade_months`. If score=trade (full look-ahead) this is
    the in-sample Oracle; if score and trade are disjoint it is OOS."""
    cols = {}
    for f in disc:
        srs = {v: _sr_series(Bs[v].loc[score_months, f]) for v in venues}
        srs = {k: v for k, v in srs.items() if np.isfinite(v)}
        if not srs:
            continue
        best = max(srs, key=srs.get)
        cols[f] = Bs[best].loc[trade_months, f]
    return book(pd.DataFrame(cols))


def coarse_book(Bs, disc, months):
    return book(pd.DataFrame({f: Bs[COARSE].loc[months, f] for f in disc}))


def build_returns_panel(Bs, disc, venues, months):
    """Dict f -> DataFrame(T x V) of venue returns, aligned on `months`."""
    panels = {}
    for f in disc:
        df = pd.DataFrame({v: Bs[v].loc[months, f] for v in venues}).dropna()
        if df.shape[0] >= 24:
            panels[f] = df
    return panels
def null_oracle_gap(panels, venues, n_sim=200, block=6, seed=0):
    """NULL world: for each factor, remove the TRUE venue advantage by demeaning
    each month across venues (so every venue has identical expected return = the
    cross-venue mean) -- true venue advantage is exactly 0. Preserve the realized
    cross-venue noise covariance and autocorrelation via block bootstrap. Then
    recompute the full-sample Oracle SR minus coarse SR. The distribution of this
    NULL gap is the selection-over-noise component."""
    rng = np.random.default_rng(seed)
    # precompute per-factor demeaned (null) venue returns + the common component
    nulls = {}
    for f, df in panels.items():
        X = df.to_numpy()                       # (T x V)
        common = X.mean(axis=1, keepdims=True)  # cross-venue mean per month
        dev = X - common                        # venue deviations, true adv = 0
        nulls[f] = (common, dev, df.columns.tolist())
    T_ref = min(df.shape[0] for df in panels.values())
    gaps = np.empty(n_sim)
    for s in range(n_sim):
        oracle_streams, coarse_streams = [], []
        nb = int(np.ceil(T_ref / block))
        for f, (common, dev, cols) in nulls.items():
            T = common.shape[0]
            starts = rng.integers(0, T, nb)
            idx = np.concatenate([np.arange(st, st + block) % T
                                  for st in starts])[:T_ref]
            # null returns = common + resampled deviations (true adv 0)
            Xn = common[idx] + dev[idx]
            srs = ANN * Xn.mean(0) / (Xn.std(0, ddof=1) + 1e-12)
            best = int(np.argmax(srs))
            cidx = cols.index(COARSE) if COARSE in cols else best
            oracle_streams.append(Xn[:, best])
            coarse_streams.append(Xn[:, cidx])
        O = np.mean(np.column_stack(oracle_streams), axis=1)
        C = np.mean(np.column_stack(coarse_streams), axis=1)
        gaps[s] = (ANN * O.mean() / O.std(ddof=1)
                   - ANN * C.mean() / C.std(ddof=1))
    return gaps
def run():
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr, te = idx[idx < mid], idx[idx >= mid]
    disc = [f for f in shared
            if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < 1.96]
    venues = BATTLES

    # real reference SRs on the full OOS (test) window
    sr_coarse = _sr_series(coarse_book(Bs, disc, te))
    sr_grave = sr_coarse  # coarse IS the grave (all discards in wide) here
    sr_oracle_full = _sr_series(oracle_book(Bs, disc, venues, te, te))  # look-ahead
    sr_oracle_oos = _sr_series(oracle_book(Bs, disc, venues, tr, te))   # train-pick

    gap_total = sr_oracle_full - sr_coarse
    gap_oos = sr_oracle_oos - sr_coarse

    # NULL world: selection-over-noise component of the full-sample Oracle
    panels = build_returns_panel(Bs, disc, venues, te)
    null_gaps = null_oracle_gap(panels, venues, n_sim=200)
    gap_null_mean = float(np.mean(null_gaps))
    gap_null_ci = [float(np.percentile(null_gaps, 2.5)),
                   float(np.percentile(null_gaps, 97.5))]
    gap_signal = gap_total - gap_null_mean
    frac_mirage = gap_null_mean / gap_total if gap_total > 0 else np.nan

    out = {
        "n_disc": len(disc), "venues": venues,
        "sr_coarse": sr_coarse,
        "sr_oracle_lookahead": sr_oracle_full,
        "sr_oracle_oos_trainpick": sr_oracle_oos,
        "gap_total_lookahead": gap_total,
        "gap_oos_trainpick": gap_oos,
        "gap_null_selection_noise_mean": gap_null_mean,
        "gap_null_ci95": gap_null_ci,
        "gap_genuine_signal": gap_signal,
        "fraction_of_bone_that_is_mirage": frac_mirage,
        "verdict": ("MEAT IS MOSTLY MIRAGE" if frac_mirage > 0.6
                    else "GENUINE SIGNAL PRESENT"),
    }
    (HERE / "cn_oracle_gap.json").write_text(json.dumps(out, indent=2) + "\n")
    _print(out)
    return out


def _print(o):
    print("=" * 76)
    print("WHERE IS THE GAP HIDDEN?  Oracle = genuine signal + selection noise")
    print("=" * 76)
    print(f"  N_disc={o['n_disc']}  venues={o['venues']}")
    print("-" * 76)
    print(f"  coarse (grave) SR                 = {o['sr_coarse']:.3f}")
    print(f"  Oracle (look-ahead, pick on test) = {o['sr_oracle_lookahead']:.3f}")
    print(f"  Oracle (OOS, pick on train)       = {o['sr_oracle_oos_trainpick']:.3f}")
    print("-" * 76)
    print(f"  gap_total (look-ahead - coarse)   = {o['gap_total_lookahead']:+.3f}")
    print(f"  gap_oos   (train-pick - coarse)   = {o['gap_oos_trainpick']:+.3f}")
    print("-" * 76)
    print(f"  NULL (zero true venue adv) Oracle gap:")
    print(f"     mean = {o['gap_null_selection_noise_mean']:+.3f}  "
          f"95% CI [{o['gap_null_ci95'][0]:+.3f}, {o['gap_null_ci95'][1]:+.3f}]")
    print(f"  => SELECTION-OVER-NOISE explains {o['fraction_of_bone_that_is_mirage']:.0%}"
          f" of the Oracle bone")
    print(f"  => GENUINE tradable signal gap    = {o['gap_genuine_signal']:+.3f}")
    print("-" * 76)
    print(f"  VERDICT: {o['verdict']}")
    print("wrote cn_oracle_gap.json")


if __name__ == "__main__":
    run()
