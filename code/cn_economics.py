"""ECONOMIC VALUE EXCAVATION.

The routing dimension is exhausted (tradable rank 0, oracle is mirage). This
module mines the WEIGHTING dimension, which does NOT require forecasting venue
advantage -- only the covariance of returns, the most persistent object in
finance. We push it to the limit and translate everything into money:

  weighting schemes on the coarse small-cap venue (wide):
    EW            equal weight (the baseline capacity, SR 1.302)
    RP            risk parity 1/sigma, frozen on train
    MV            minimum variance (shrinkage covariance)
    HRP           hierarchical risk parity (cluster -> allocate)
    MV_longonly   min-variance with no-short constraint
  cross-venue risk allocation:
    RP across the four venue books (not within wide)

  money metrics for each:
    annualized mean, vol, Sharpe
    6-factor alpha (FF5+Mom) and t-stat
    certainty-equivalent gain at gamma=5 vs EW
    turnover and net Sharpe at 100 bps
    capacity proxy: how concentrated are the weights (eff. N, max weight)
    decade stability

  distinguish RP from the plain low-volatility anomaly: does RP still add value
  AFTER controlling for the BAB/low-vol factor exposure?

Writes cn_economics.json.
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
KF = Path(os.environ.get("LOCALSTRENGTH_DATA", "E:/BOAT-FX-DATA")) / "ken_french"
ANN = np.sqrt(12.0)
COARSE = "wide"

def _sr(x):
    x = pd.Series(x).dropna()
    if x.size < 12 or x.std(ddof=1) == 0:
        return np.nan
    return float(ANN * x.mean() / x.std(ddof=1))


def shrink_cov(R, delta=None):
    """Ledoit-Wolf-style shrinkage toward a diagonal target."""
    S = np.cov(R, rowvar=False)
    p = S.shape[0]
    mu = np.trace(S) / p
    target = mu * np.eye(p)
    if delta is None:
        delta = 0.3
    return (1 - delta) * S + delta * target


def w_equal(R):
    n = R.shape[1]
    return np.ones(n) / n


def w_riskparity(R):
    vol = R.std(axis=0, ddof=1)
    vol[vol == 0] = np.nan
    w = 1.0 / vol
    w = np.nan_to_num(w)
    return w / w.sum()


def w_minvar(R):
    S = shrink_cov(R)
    try:
        inv = np.linalg.pinv(S)
        ones = np.ones(S.shape[0])
        w = inv @ ones
        w = w / w.sum()
    except Exception:
        return w_equal(R)
    return w


def w_minvar_long(R):
    """Min-variance with no-short via simple projected gradient."""
    S = shrink_cov(R)
    n = S.shape[0]
    w = np.ones(n) / n
    for _ in range(500):
        grad = S @ w
        w = w - 0.01 * grad
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            w = np.ones(n) / n
        w = w / w.sum()
    return w


def w_hrp(R):
    """Hierarchical risk parity: cluster by correlation distance, recursive
    bisection allocation."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    S = np.cov(R, rowvar=False)
    std = np.sqrt(np.diag(S))
    std[std == 0] = 1e-9
    C = S / np.outer(std, std)
    C = np.clip(C, -1, 1)
    D = np.sqrt(0.5 * (1 - C))
    np.fill_diagonal(D, 0.0)
    try:
        link = linkage(squareform(D, checks=False), method="single")
        order = leaves_list(link)
    except Exception:
        order = np.arange(R.shape[1])
    w = np.ones(R.shape[1])

    def bisect(items):
        if len(items) == 1:
            return
        half = len(items) // 2
        left, right = items[:half], items[half:]
        for side in (left, right):
            cov = S[np.ix_(side, side)]
            iv = 1.0 / np.diag(cov)
            iv = iv / iv.sum()
            var = iv @ cov @ iv
            side_var = var
        # cluster variances
        def cvar(side):
            cov = S[np.ix_(side, side)]
            iv = 1.0 / np.diag(cov)
            iv = iv / iv.sum()
            return iv @ cov @ iv
        vl, vr = cvar(left), cvar(right)
        alpha = 1 - vl / (vl + vr)
        for i in left:
            w[i] *= alpha
        for i in right:
            w[i] *= (1 - alpha)
        bisect(left)
        bisect(right)

    bisect(list(order))
    return w / w.sum()
def load_factors6():
    ff = pd.read_csv(KF / "FF5_monthly.csv")
    ff["Date"] = pd.to_datetime(ff["Date"].astype(str), format="%Y%m")
    ff = ff.set_index("Date")
    ff.index = ff.index.to_period("M").to_timestamp("M")
    mom = pd.read_csv(KF / "Momentum_monthly.csv")
    mom["Date"] = pd.to_datetime(mom["Date"].astype(str), format="%Y%m")
    mom = mom.set_index("Date")
    mom.index = mom.index.to_period("M").to_timestamp("M")
    df = ff.join(mom, how="inner")   # keep PERCENT units to match OSAP returns
    return df[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]]


def alpha6(book, F6):
    df = pd.concat([book.rename("y"), F6], axis=1).dropna()
    if len(df) < 36:
        return None
    y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df[F6.columns].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(s2 * XtX_inv))
    a_m = beta[0]
    t = a_m / se[0]
    # OSAP returns are already in PERCENT units; annualize by *12 only.
    return {"alpha_ann_pct": float(a_m * 12), "t_alpha": float(t),
            "beta_mkt": float(beta[1])}


def cer_gain(book, bench, gamma=5.0):
    """Certainty-equivalent return gain (annualized, %) of book over bench at CRRA
    gamma on a mean-variance investor: CER = mean - 0.5*gamma*var (monthly), ann."""
    b = pd.concat([book.rename("b"), bench.rename("c")], axis=1).dropna()
    # returns in percent; convert to decimal for CRRA mean-variance, then to %/yr
    def cer(x):
        xd = x / 100.0
        return xd.mean() - 0.5 * gamma * xd.var(ddof=1)
    return float((cer(b["b"]) - cer(b["c"])) * 12 * 100)


def turnover_cost(weights_series, R, bps):
    """Approximate: frozen weights -> turnover only from drift re-set monthly.
    weights_series is the fixed weight vector. We assume monthly rebalance back
    to target; one-way turnover ~ sum|w_target - w_drifted|."""
    w = weights_series
    to = 0.0
    prev = w.copy()
    months = 0
    for t in range(1, R.shape[0]):
        r = R[t - 1]
        drift = prev * (1 + r)
        drift = drift / drift.sum()
        to += np.abs(w - drift).sum()
        prev = w.copy()
        months += 1
    avg_oneway = to / max(months, 1) / 2
    drag_monthly = (bps / 1e4) * avg_oneway
    return float(avg_oneway), float(drag_monthly)


def eff_n(w):
    w = w[w > 0]
    return float(1.0 / np.sum(w ** 2)) if w.size else np.nan
def run():
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr, te = idx[idx < mid], idx[idx >= mid]
    disc = [f for f in shared
            if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < 1.96]
    F6 = load_factors6()

    # discard return panel in the coarse small-cap venue, train & test
    Rtr = Bs[COARSE].loc[tr, disc].dropna(axis=1, how="all")
    cols = [c for c in disc if c in Rtr.columns]
    Rtr = Bs[COARSE].loc[tr, cols].fillna(0.0).to_numpy()
    Rte_df = Bs[COARSE].loc[te, cols].fillna(0.0)
    Rte = Rte_df.to_numpy()

    schemes = {"EW": w_equal, "RP": w_riskparity, "MV": w_minvar,
               "MV_long": w_minvar_long, "HRP": w_hrp}
    results = {}
    weights = {}
    for name, fn in schemes.items():
        w = fn(Rtr)                              # weights frozen on TRAIN
        weights[name] = w
        book = pd.Series(Rte @ w, index=Rte_df.index)
        results[name] = book

    ew_book = results["EW"]
    out = {"n_disc": len(cols), "schemes": {}}
    for name, book in results.items():
        w = weights[name]
        a = alpha6(book, F6)
        avg_to, drag = turnover_cost(w, Rte, 100)
        net = book - drag
        rec = {
            "sharpe_gross": _sr(book),
            "sharpe_net_100bps": _sr(net),
            "mean_ann_pct": float(book.mean() * 12),
            "vol_ann_pct": float(book.std(ddof=1) * np.sqrt(12)),
            "alpha6_ann_pct": a["alpha_ann_pct"] if a else None,
            "t_alpha6": a["t_alpha"] if a else None,
            "cer_gain_vs_ew_ann_pct": cer_gain(book, ew_book),
            "turnover_oneway_monthly": avg_to,
            "eff_n": eff_n(w), "max_weight": float(w.max()),
        }
        # decade gaps vs EW
        both = pd.concat([book.rename("b"), ew_book.rename("e")], axis=1).dropna()
        dec = {}
        for d, g in both.groupby(both.index.year // 10 * 10):
            if len(g) >= 24:
                dec[str(int(d)) + "s"] = round(_sr(g["b"]) - _sr(g["e"]), 3)
        rec["decade_gap_vs_ew"] = dec
        out["schemes"][name] = rec

    # cross-venue risk allocation: RP across the four venue EW books
    venue_books = {}
    for v in BATTLES:
        venue_books[v] = Bs[v].loc[te, cols].fillna(0.0).mean(axis=1)
    VB = pd.DataFrame(venue_books).dropna()
    VBtr = pd.DataFrame({v: Bs[v].loc[tr, cols].fillna(0.0).mean(axis=1)
                         for v in BATTLES}).dropna()
    wv = w_riskparity(VBtr.to_numpy())
    cross_book = pd.Series(VB.to_numpy() @ wv, index=VB.index)
    out["cross_venue_rp"] = {
        "sharpe": _sr(cross_book),
        "weights": {v: float(wv[i]) for i, v in enumerate(BATTLES)},
        "vs_coarse": _sr(cross_book) - _sr(ew_book)}

    # is RP just the low-vol anomaly? regress RP-minus-EW on EW itself + factors
    rp_minus_ew = (results["RP"] - ew_book).dropna()
    a_spread = alpha6(rp_minus_ew, F6)
    out["rp_minus_ew_is_alpha"] = {
        "mean_ann_pct": float(rp_minus_ew.mean() * 12 * 100),
        "alpha6_ann_pct": a_spread["alpha_ann_pct"] if a_spread else None,
        "t_alpha6": a_spread["t_alpha"] if a_spread else None}
    # control for the low-volatility anomaly: build a BAB-like factor from the
    # discards (low-vol minus high-vol, equal weight) on the TEST window, add to
    # the 6 factors, and see if MV/RP spreads still carry alpha.
    vol_tr = Rtr.std(0, ddof=1)
    lo = np.argsort(vol_tr)[:len(cols) // 3]
    hi = np.argsort(vol_tr)[-len(cols) // 3:]
    bab = pd.Series(Rte[:, lo].mean(1) - Rte[:, hi].mean(1), index=Rte_df.index)
    F7 = pd.concat([F6, bab.rename("BAB")], axis=1).dropna()
    for nm in ("RP", "MV", "HRP"):
        spread = (results[nm] - ew_book).dropna()
        a7 = alpha6(spread, F7)
        out["schemes"][nm]["alpha_vs_6F_plus_BAB"] = (
            {"alpha_ann_pct": a7["alpha_ann_pct"], "t": a7["t_alpha"]}
            if a7 else None)
    (HERE / "cn_economics.json").write_text(json.dumps(out, indent=2,
                                                       default=str) + "\n")
    _print(out)
    return out


def _print(o):
    print("=" * 90)
    print("ECONOMIC VALUE EXCAVATION  (weighting dimension, discards in wide)")
    print("=" * 90)
    print(f"  N_disc={o['n_disc']}")
    h = f"  {'scheme':9s} {'SR':>6s} {'SRnet':>6s} {'mean%':>6s} {'vol%':>6s} " \
        f"{'a6%':>6s} {'t_a':>5s} {'CERvsEW':>7s} {'effN':>5s} {'maxw':>5s}"
    print(h)
    for name, r in o["schemes"].items():
        print(f"  {name:9s} {r['sharpe_gross']:6.3f} {r['sharpe_net_100bps']:6.3f} "
              f"{r['mean_ann_pct']:6.1f} {r['vol_ann_pct']:6.1f} "
              f"{(r['alpha6_ann_pct'] or 0):6.2f} {(r['t_alpha6'] or 0):5.2f} "
              f"{r['cer_gain_vs_ew_ann_pct']:7.2f} {r['eff_n']:5.1f} "
              f"{r['max_weight']:5.2f}")
    print("-" * 90)
    for name, r in o["schemes"].items():
        print(f"  {name:9s} decade gap vs EW: {r['decade_gap_vs_ew']}")
    print("-" * 90)
    cv = o["cross_venue_rp"]
    print(f"  cross-venue RP: SR={cv['sharpe']:.3f}  vs_coarse={cv['vs_coarse']:+.3f}"
          f"  weights={ {k: round(v,2) for k,v in cv['weights'].items()} }")
    rm = o["rp_minus_ew_is_alpha"]
    print(f"  RP-minus-EW spread: mean={rm['mean_ann_pct']:.2f}%/yr  "
          f"6-factor alpha={rm['alpha6_ann_pct']:.2f}%/yr  t={rm['t_alpha6']:.2f}")
    print("  --- alpha AFTER controlling for low-vol (BAB) factor ---")
    for nm in ("RP", "MV", "HRP"):
        a7 = o["schemes"][nm].get("alpha_vs_6F_plus_BAB")
        if a7:
            print(f"     {nm}-EW spread vs 6F+BAB: alpha={a7['alpha_ann_pct']:.2f}"
                  f"%/yr  t={a7['t']:.2f}")
    print("wrote cn_economics.json")


if __name__ == "__main__":
    run()
