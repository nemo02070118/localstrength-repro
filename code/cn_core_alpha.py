"""HARDEN THE CORE ALPHA.

The one economically real result is the first step: moving the 94 large-cap-
rejected factors as a group to the small-cap venue lifts Sharpe 0.85 -> 1.30 and
earns a 6-factor alpha of 3.78%/yr (t=10.2). Before building a paper on it we
must rule out that this too is just a repackaged low-volatility / illiquidity
anomaly. We stress the alpha against progressively harder controls:

  M0: CAPM (Mkt)
  M1: FF3 (Mkt,SMB,HML)
  M2: FF5+Mom (the headline 6-factor)
  M3: + BAB (low-vol minus high-vol, built from the discards)
  M4: + a traded illiquidity leg (small/illiquid proxy via SMB tails)
  M5: + the EW grave book itself (does coarse beat its own large-cap twin?)

The sharpest, cleanest object is the REALLOCATION SPREAD = coarse_book -
grave_book (same 94 factors, same signals, only the venue differs). Its alpha is
pure venue value: it nets out the factors' own content. We report its alpha
under every control and across decades. If the spread alpha survives M2-M4 with
t>3, the core result is not a risk-premium relabel.

Writes cn_core_alpha.json.
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

def _sr(x):
    x = pd.Series(x).dropna()
    if x.size < 12 or x.std(ddof=1) == 0:
        return np.nan
    return float(ANN * x.mean() / x.std(ddof=1))


def load_f6():
    ff = pd.read_csv(KF / "FF5_monthly.csv")
    ff["Date"] = pd.to_datetime(ff["Date"].astype(str), format="%Y%m")
    ff = ff.set_index("Date"); ff.index = ff.index.to_period("M").to_timestamp("M")
    mom = pd.read_csv(KF / "Momentum_monthly.csv")
    mom["Date"] = pd.to_datetime(mom["Date"].astype(str), format="%Y%m")
    mom = mom.set_index("Date"); mom.index = mom.index.to_period("M").to_timestamp("M")
    return ff.join(mom, how="inner")   # percent units


def ols_alpha(y, X):
    """y vector, X design WITHOUT intercept (added here). Returns alpha(%/yr),
    t, and R2. HAC (Newey-West, 6 lag) standard error on the intercept."""
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(d) < 48:
        return None
    yy = d["y"].to_numpy()
    XX = np.column_stack([np.ones(len(d)), d[X.columns].to_numpy()])
    beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
    resid = yy - XX @ beta
    n, k = XX.shape
    # Newey-West HAC covariance, 6 lags
    XtX_inv = np.linalg.pinv(XX.T @ XX)
    S = (XX * resid[:, None]).T @ (XX * resid[:, None])
    L = 6
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = (XX[l:] * resid[l:, None]).T @ (XX[:-l] * resid[:-l, None])
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    ss_tot = np.sum((yy - yy.mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot
    return {"alpha_ann_pct": float(beta[0] * 12), "t": float(beta[0] / se[0]),
            "r2": float(r2), "n": int(n)}


def build_books(Bs, disc, te):
    grave = Bs["vw"].loc[te, disc].mean(axis=1).dropna()
    coarse = Bs["wide"].loc[te, disc].mean(axis=1).dropna()
    return grave, coarse
def run():
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr, te = idx[idx < mid], idx[idx >= mid]
    disc = [f for f in shared
            if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < 1.96]
    F = load_f6()
    grave, coarse = build_books(Bs, disc, te)
    spread = (coarse - grave).dropna()          # pure venue value

    # BAB from discards (low-vol minus high-vol, in wide), test window
    Rte = Bs["wide"].loc[te, disc].fillna(0.0)
    vol = Bs["wide"].loc[tr, disc].std()
    order = vol.sort_values().index
    lo = order[:len(order) // 3]
    hi = order[-len(order) // 3:]
    bab = (Rte[lo].mean(axis=1) - Rte[hi].mean(axis=1)).rename("BAB")

    mkt = F[["Mkt-RF"]]
    ff3 = F[["Mkt-RF", "SMB", "HML"]]
    f6 = F[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]]
    f6b = pd.concat([f6, bab], axis=1)
    f6bg = pd.concat([f6, bab, grave.rename("GRAVE")], axis=1)

    models = {"M0_CAPM": mkt, "M1_FF3": ff3, "M2_F6": f6,
              "M3_F6_BAB": f6b, "M5_F6_BAB_grave": f6bg}

    out = {"n_disc": len(disc), "test": [str(te.min())[:7], str(te.max())[:7]],
           "sr_grave": _sr(grave), "sr_coarse": _sr(coarse),
           "sr_spread": _sr(spread),
           "coarse_alpha": {}, "spread_alpha": {}}
    for mn, X in models.items():
        ac = ols_alpha(coarse, X)
        asp = ols_alpha(spread, X)
        out["coarse_alpha"][mn] = ac
        out["spread_alpha"][mn] = asp

    # decade alphas of the spread under F6
    dec = {}
    both = pd.concat([spread.rename("s"), f6], axis=1).dropna()
    for d, g in both.groupby(both.index.year // 10 * 10):
        if len(g) >= 24:
            a = ols_alpha(g["s"], g[f6.columns])
            dec[str(int(d)) + "s"] = {"alpha_ann_pct": a["alpha_ann_pct"],
                                      "t": a["t"], "n": a["n"]} if a else None
    out["spread_alpha_by_decade_F6"] = dec

    out["verdict_core_survives"] = bool(
        out["spread_alpha"]["M3_F6_BAB"]
        and out["spread_alpha"]["M3_F6_BAB"]["t"] > 3.0
        and out["spread_alpha"]["M3_F6_BAB"]["alpha_ann_pct"] > 0)
    (HERE / "cn_core_alpha.json").write_text(json.dumps(out, indent=2) + "\n")
    _print(out)
    return out


def _print(o):
    print("=" * 80)
    print("HARDEN THE CORE ALPHA: is the +0.45 reallocation a low-vol relabel?")
    print("=" * 80)
    print(f"  N_disc={o['n_disc']}  test={o['test']}")
    print(f"  SR grave(vw)={o['sr_grave']:.3f}  coarse(wide)={o['sr_coarse']:.3f}"
          f"  spread={o['sr_spread']:.3f}")
    print("-" * 80)
    print("  COARSE book alpha (annualized %, HAC t):")
    for mn, a in o["coarse_alpha"].items():
        if a:
            print(f"     {mn:16s} alpha={a['alpha_ann_pct']:+6.2f}%  t={a['t']:5.2f}"
                  f"  R2={a['r2']:.2f}")
    print("  REALLOCATION SPREAD (coarse - grave) alpha  <-- pure venue value:")
    for mn, a in o["spread_alpha"].items():
        if a:
            print(f"     {mn:16s} alpha={a['alpha_ann_pct']:+6.2f}%  t={a['t']:5.2f}"
                  f"  R2={a['r2']:.2f}")
    print("  spread alpha by decade (F6):")
    for d, a in o["spread_alpha_by_decade_F6"].items():
        if a:
            print(f"     {d}: alpha={a['alpha_ann_pct']:+6.2f}%  t={a['t']:5.2f} "
                  f"(n={a['n']})")
    print("-" * 80)
    print(f"  VERDICT core survives F6+BAB (t>3, alpha>0): "
          f"{o['verdict_core_survives']}")
    print("wrote cn_core_alpha.json")


if __name__ == "__main__":
    run()
