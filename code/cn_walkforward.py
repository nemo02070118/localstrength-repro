"""
cn_walkforward.py
=================
True walk-forward venue routing for BOTH markets (plan: walkforward).

Instead of a single train/test split, we re-select each factor's venue (fine)
and the whole-book venue (coarse) EVERY YEAR using ONLY data available up to that
point (expanding window, min 60 months), then trade the next 12 months. This is
the strongest robustness test: no look-ahead in the routing decision at all.

US  : OSAP long-short panels (vw/nyse/me20/wide); universal=vw, coarse=wide.
A-sh: QuantaAlpha 5441-stock panel; long-only; venues small/mid/large;
      universal=all; coarse=train-best single venue.

Writes cn_walkforward.json.
"""
import json, os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "cn_walkforward.json"
ANN = np.sqrt(12.0)


def sr(x):
    x = pd.Series(x).dropna()
    if x.size < 12 or x.std(ddof=1) == 0:
        return np.nan
    return float(ANN * x.mean() / x.std(ddof=1))


# ======================================================================= US
def walkforward_us():
    from cn_resurrection import load_battlefields, _tstat, BATTLES
    Bs, shared, idx = load_battlefields()
    idx = pd.DatetimeIndex(idx)
    # sign-fix burns the first 60m to NaN; valid data starts ~month 61.
    valid = Bs["vw"].dropna(how="all").index
    idx = idx[idx >= valid.min()]
    # discarded in vw over the FIRST 60 valid months (fixed universe)
    warm = idx[:60]
    disc = [f for f in shared
            if np.isfinite(_tstat(Bs["vw"].loc[warm, f]))
            and abs(_tstat(Bs["vw"].loc[warm, f])) < 1.96]
    years = sorted(set(idx.year))
    rows_u, rows_c, rows_f = [], [], []
    for y in years:
        past = idx[idx.year < y]
        cur = idx[idx.year == y]
        if len(past) < 60 or len(cur) == 0:
            continue
        # fine: each factor's past-best venue; coarse fixed = wide (broad universe)
        route = {}
        for f in disc:
            s = {v: sr(Bs[v].loc[past, f]) for v in BATTLES}
            route[f] = max(BATTLES, key=lambda v: s[v] if np.isfinite(s[v]) else -9)
        for dt in cur:
            u = np.nanmean([Bs["vw"].loc[dt, f] for f in disc])
            c = np.nanmean([Bs["wide"].loc[dt, f] for f in disc])
            fx = np.nanmean([Bs[route[f]].loc[dt, f] for f in disc])
            rows_u.append((dt, u)); rows_c.append((dt, c)); rows_f.append((dt, fx))
    U = pd.Series(dict(rows_u)); C = pd.Series(dict(rows_c)); Fx = pd.Series(dict(rows_f))
    return {"n_disc": len(disc),
            "span": [str(U.index.min())[:7], str(U.index.max())[:7]],
            "sharpe": {"universal": sr(U), "coarse_wide": sr(C), "fine": sr(Fx)},
            "lift_coarse_vs_universal": round(sr(C) - sr(U), 3),
            "lift_fine_vs_universal": round(sr(Fx) - sr(U), 3),
            "lift_fine_vs_coarse": round(sr(Fx) - sr(C), 3)}


# ================================================================== A-share
def walkforward_ashare():
    import cn_ashare_salvage as S
    d = S.load_monthly(); m = S.build_monthly(d); del d
    R = m["ret_m"].unstack("inst").sort_index()
    DV = m["dvol"].unstack("inst").sort_index()
    VOLd = m["vol_d"].unstack("inst").reindex_like(R)
    MAXd = m["max_d"].unstack("inst").reindex_like(R)
    SIG = {"REV": -R, "MOM": R.rolling(12).sum().shift(1),
           "VOL": -VOLd, "MAX": -MAXd, "TURN": -DV.pct_change()}
    facs = list(SIG)
    sv = ["small", "mid", "large"]
    FLOOR, TCOST = 0.30, 0.001
    liq = DV.rank(axis=1, pct=True); trad = liq > FLOOR
    Rt = R.where(trad); VEN = S.venue_masks(DV.where(trad))
    venues = sv + ["all"]
    series = {}
    for f in facs:
        for v in venues:
            vm = None if v == "all" else VEN[v]
            series[(f, v)] = S.ls_return(SIG[f], Rt, vm, tcost=TCOST, long_only=True)
    series = {k: pd.Series(v) for k, v in series.items()}
    idx = R.index
    years = sorted(set(idx.year))
    rows_u, rows_c, rows_f = [], [], []
    for y in years:
        past = idx[idx.year < y]; cur = idx[idx.year == y]
        if len(past) < 36 or len(cur) == 0:
            continue
        route = {}
        for f in facs:
            s = {v: sr(series[(f, v)].reindex(past)) for v in sv}
            route[f] = max(sv, key=lambda v: s[v] if np.isfinite(s[v]) else -9)
        coarse_s = {v: sr(pd.concat([series[(f, v)].reindex(past) for f in facs],
                                    axis=1).mean(axis=1)) for v in sv}
        cv = max(sv, key=lambda v: coarse_s[v] if np.isfinite(coarse_s[v]) else -9)
        for dt in cur:
            u = np.nanmean([series[(f, "all")].get(dt, np.nan) for f in facs])
            c = np.nanmean([series[(f, cv)].get(dt, np.nan) for f in facs])
            fx = np.nanmean([series[(f, route[f])].get(dt, np.nan) for f in facs])
            rows_u.append((dt, u)); rows_c.append((dt, c)); rows_f.append((dt, fx))
    U = pd.Series(dict(rows_u)); C = pd.Series(dict(rows_c)); Fx = pd.Series(dict(rows_f))
    return {"n_factors": len(facs), "venues": venues,
            "span": [str(U.index.min())[:7], str(U.index.max())[:7]],
            "sharpe": {"universal": sr(U), "coarse": sr(C), "fine": sr(Fx)},
            "lift_coarse_vs_universal": round(sr(C) - sr(U), 3),
            "lift_fine_vs_universal": round(sr(Fx) - sr(U), 3),
            "lift_fine_vs_coarse": round(sr(Fx) - sr(C), 3)}


if __name__ == "__main__":
    print("[US] walk-forward ...")
    us = walkforward_us()
    print("  US:", us["sharpe"], "| coarse-univ", us["lift_coarse_vs_universal"],
          "| fine-coarse", us["lift_fine_vs_coarse"])
    print("[A-share] walk-forward ...")
    ash = walkforward_ashare()
    print("  A:", ash["sharpe"], "| coarse-univ", ash["lift_coarse_vs_universal"],
          "| fine-coarse", ash["lift_fine_vs_coarse"])
    out = {"method": "annual walk-forward: expanding-window venue selection, "
                     "trade next 12m; no look-ahead in routing",
           "US": us, "A_share": ash}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT)
