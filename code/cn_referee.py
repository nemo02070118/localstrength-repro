"""Adversarial referee battery for the Local-Strength paper. Answers the kill
shots with data: risk-adjustment (is it just SMB?), transaction costs (does it
survive?), split-robustness (is 1975 cherry-picked?), block/stationary
bootstrap (serial correlation), Newey-West, deflated Sharpe, sub-periods, and
the canon-spanning test (does revived add alpha beyond the surviving factors?).

Writes cn_referee.json.
"""
from __future__ import annotations

import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cn_resurrection import load_battlefields, _tstat, _sr, _book, BATTLES

warnings.filterwarnings("ignore")
HERE = Path(os.environ.get("LOCALSTRENGTH_RESULTS", str(Path(__file__).resolve().parent.parent / "results")))
DATA = Path(os.environ.get("LOCALSTRENGTH_DATA", "E:/BOAT-FX-DATA"))
ANN = np.sqrt(12.0)


def _ff():
    ff = pd.read_csv(DATA / "ken_french" / "FF5_monthly.csv")
    mom = pd.read_csv(DATA / "ken_french" / "Momentum_monthly.csv")
    ff["Date"] = pd.to_datetime(ff["Date"].astype(str), format="%Y%m")
    mom["Date"] = pd.to_datetime(mom["Date"].astype(str), format="%Y%m")
    ff = ff.set_index("Date") / 100.0
    mom = mom.set_index("Date") / 100.0
    ff = ff.join(mom, how="left")
    ff.index = ff.index + pd.offsets.MonthEnd(0)
    return ff


def _nw_t(y, X=None, lags=12):
    """Newey-West t-stat of the mean (or intercept if X given)."""
    y = np.asarray(y, float)
    if X is None:
        X = np.ones((len(y), 1))
    else:
        X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (resid[:, None] * X).T @ (resid[:, None] * X)
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = (resid[l:, None] * X[l:]).T @ (resid[:-l, None] * X[:-l])
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return b, b / se


def _books(Bs, shared, tr, te, hurdle=1.96):
    disc = [f for f in shared if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < hurdle]
    keep = [f for f in shared if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) >= hurdle]
    pool = {n: np.nanmean([_sr(Bs[n].loc[tr, f]) for f in disc]) for n in BATTLES}
    cb = max(pool, key=pool.get)
    _, mg = _book(Bs, disc, {f: "vw" for f in disc}, te)
    _, mc = _book(Bs, disc, {f: cb for f in disc}, te)
    _, mk = _book(Bs, keep, {f: "vw" for f in keep}, te)
    return disc, keep, cb, mg, mc, mk


def run():
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr = idx[idx < mid]; te = idx[idx >= mid]
    disc, keep, cb, mg, mc, mk = _books(Bs, shared, tr, te)
    gain = (mc - mg).dropna()                      # the coarse improvement stream
    out = {"n_disc": len(disc), "n_keep": len(keep), "coarse_bf": cb}

    # ---- B. risk adjustment: is the gain just SMB / market? ----
    ff = _ff()
    # OSAP LS returns are in PERCENT units; convert to decimal to match FF.
    mc = mc / 100.0; mg = mg / 100.0; mk = mk / 100.0
    gain = (mc - mg).dropna()
    both = pd.concat([mc.rename("coarse"), mg.rename("grave"),
                      gain.rename("gain")], axis=1).join(ff, how="inner").dropna()
    facs = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    risk = {}
    for book in ("coarse", "gain"):
        y = both[book].to_numpy()
        for model, cols in (("CAPM", ["Mkt-RF"]),
                            ("FF3", ["Mkt-RF", "SMB", "HML"]),
                            ("FF5", ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]),
                            ("FF6", facs)):
            b, t = _nw_t(y, both[cols].to_numpy(), lags=12)
            risk[f"{book}_{model}"] = {
                "alpha_ann": float(b[0] * 12), "alpha_t": float(t[0]),
                "betas": {c: float(bi) for c, bi in zip(cols, b[1:])}}
    out["risk_adjustment"] = risk

    # ---- C. transaction costs (turnover from monthly weight flips) ----
    # LS factor returns already net of holding; approximate turnover by sign of
    # each factor's monthly return change is not available -> use book vol proxy:
    # charge cost on gross exposure assuming full monthly rebalance of N names.
    # Costs: reallocation across battlefields is a one-time (annual-rebalance)
    # decision; within-universe factor turnover is identical to the grave book,
    # so the incremental cost falls on the GAIN at the annual reallocation only.
    cost_coarse, cost_gain = {}, {}
    for bps in (0, 10, 30, 50, 100):
        drag = (bps / 1e4) * 2.0            # gross LS legs, once per year
        nc = mc.copy(); nc.iloc[::12] -= drag
        ng = gain.copy(); ng.iloc[::12] -= drag
        cost_coarse[str(bps)] = float(_sr(nc.dropna()))
        cost_gain[str(bps)] = float(_sr(ng.dropna()))
    out["cost_net_sr_coarse_annual"] = cost_coarse
    out["cost_net_sr_gain_annual"] = cost_gain
    out["breakeven_bps_oneway_gain"] = float(gain.mean() * 12 / 2.0 * 1e4)

    # ---- D. serial-correlation-robust inference on the gain ----
    g = gain.to_numpy()
    _, tnw = _nw_t(g, lags=12)
    out["gain_mean_ann"] = float(g.mean() * 12)
    out["gain_newey_west_t"] = float(tnw[0])
    # stationary bootstrap (Politis-Romano), mean block 24m
    rng = np.random.default_rng(0); n = len(g); p = 1 / 24.0
    bs = []
    for _ in range(5000):
        out_idx = []; i = rng.integers(0, n)
        while len(out_idx) < n:
            out_idx.append(i)
            if rng.random() < p:
                i = rng.integers(0, n)
            else:
                i = (i + 1) % n
        gb = g[np.array(out_idx)]
        bs.append(ANN * gb.mean() / gb.std(ddof=1))
    out["gain_stationary_boot_sr"] = {
        "median": float(np.median(bs)),
        "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "p_gt0": float(np.mean(np.array(bs) > 0))}
    # deflated Sharpe (approx, accounting for the 4 hurdle trials)
    sr = _sr(gain); T = len(g)
    sr0 = np.sqrt(2 * np.log(4)) / np.sqrt(T) * ANN   # expected max under null
    out["deflated_sharpe"] = {"sr_gain": float(sr), "haircut_null": float(sr0),
                              "survives": bool(sr > sr0)}

    # ---- G. split-year robustness ----
    split = {}
    for yr in (1960, 1970, 1980, 1990, 2000):
        cut = pd.Timestamp(f"{yr}-01-01")
        tr2 = idx[idx < cut]; te2 = idx[idx >= cut]
        if len(tr2) < 120 or len(te2) < 120:
            continue
        d2, k2, cb2, g2, c2, _ = _books(Bs, shared, tr2, te2)
        split[str(yr)] = {"coarse_bf": cb2, "sr_grave": _sr(g2),
                          "sr_coarse": _sr(c2), "gain": _sr(c2) - _sr(g2),
                          "n_disc": len(d2)}
    out["split_year_robustness"] = split

    # ---- H. canon spanning: does revived add alpha beyond the canon book? ----
    cj = pd.concat([mc.rename("rev"), mk.rename("canon")], axis=1).dropna()
    b, t = _nw_t(cj["rev"].to_numpy(), cj["canon"].to_numpy().reshape(-1, 1), 12)
    out["canon_spanning"] = {"alpha_ann": float(b[0] * 12), "alpha_t": float(t[0]),
                             "beta_canon": float(b[1])}

    (HERE / "cn_referee.json").write_text(json.dumps(out, indent=2) + "\n")
    _print(out)
    return out


def _print(o):
    print("=" * 74)
    print("REFEREE BATTERY")
    print("=" * 74)
    print(f"  discards={o['n_disc']} kept={o['n_keep']} coarse_bf={o['coarse_bf']}")
    print("-- B. risk-adjusted alpha (Newey-West t, 12 lags) --")
    for k, v in o["risk_adjustment"].items():
        print(f"   {k:14s}: alpha={v['alpha_ann']*100:+6.2f}%/yr  t={v['alpha_t']:+.2f}")
    print("-- C. transaction cost (annual reallocation; coarse / gain net SR) --")
    for bps in o["cost_net_sr_coarse_annual"]:
        print(f"   {bps:>3} bps: coarse SR={o['cost_net_sr_coarse_annual'][bps]:+.3f}"
              f"   gain SR={o['cost_net_sr_gain_annual'][bps]:+.3f}")
    print(f"   gain breakeven one-way = {o['breakeven_bps_oneway_gain']:.0f} bps")
    print("-- D. serial-robust inference on gain --")
    print(f"   gain={o['gain_mean_ann']*100:+.2f}%/yr  NW-t={o['gain_newey_west_t']:+.2f}")
    sb = o["gain_stationary_boot_sr"]
    print(f"   stationary-boot SR median={sb['median']:+.3f} CI{sb['ci95']} p>0={sb['p_gt0']:.3f}")
    ds = o["deflated_sharpe"]
    print(f"   deflated Sharpe: gain SR={ds['sr_gain']:+.3f} > null {ds['haircut_null']:+.3f}? {ds['survives']}")
    print("-- G. split-year robustness --")
    for yr, v in o["split_year_robustness"].items():
        print(f"   split {yr}: grave={v['sr_grave']:+.3f} coarse={v['sr_coarse']:+.3f} "
              f"gain={v['gain']:+.3f} (bf={v['coarse_bf']}, n={v['n_disc']})")
    cs = o["canon_spanning"]
    print(f"-- H. canon spanning: revived alpha vs canon = {cs['alpha_ann']*100:+.2f}%/yr "
          f"t={cs['alpha_t']:+.2f} (beta={cs['beta_canon']:+.2f})")
    print("wrote cn_referee.json")


if __name__ == "__main__":
    run()
