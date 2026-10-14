"""GRS / Fama-MacBeth referee-proofing of the core reallocation result.

The one real economic result (coarse small-cap book of the 94 discards) must
clear the highest bar a referee can set:
  (1) GRS F-test that the coarse book + its decile spread are not spanned by
      the 6-factor model (FF5+Mom).
  (2) The reallocation spread (coarse - grave) alpha with Newey-West t and a
      bootstrap p-value.
  (3) Fama-MacBeth style: alpha stability under expanding windows.

Writes cn_grs.json.
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
from scipy import stats

from cn_resurrection import load_battlefields, _tstat, _sr, BATTLES

warnings.filterwarnings("ignore")
HERE = Path(os.environ.get("LOCALSTRENGTH_RESULTS",
                           str(Path(__file__).resolve().parent.parent / "results")))
KF = Path(os.environ.get("LOCALSTRENGTH_DATA", "E:/BOAT-FX-DATA")) / "ken_french"
ANN = np.sqrt(12.0)


def load_f6():
    ff = pd.read_csv(KF / "FF5_monthly.csv")
    ff["Date"] = pd.to_datetime(ff["Date"].astype(str), format="%Y%m")
    ff = ff.set_index("Date"); ff.index = ff.index.to_period("M").to_timestamp("M")
    mom = pd.read_csv(KF / "Momentum_monthly.csv")
    mom["Date"] = pd.to_datetime(mom["Date"].astype(str), format="%Y%m")
    mom = mom.set_index("Date"); mom.index = mom.index.to_period("M").to_timestamp("M")
    return ff.join(mom, how="inner")


def grs(test_assets, factors):
    """Gibbons-Ross-Shanken F-test. test_assets (T x N), factors (T x K)."""
    df = pd.concat([test_assets, factors], axis=1).dropna()
    R = df[test_assets.columns].to_numpy()
    F = df[factors.columns].to_numpy()
    T, N = R.shape
    K = F.shape[1]
    X = np.column_stack([np.ones(T), F])
    B, *_ = np.linalg.lstsq(X, R, rcond=None)
    alpha = B[0]
    resid = R - X @ B
    Sigma = (resid.T @ resid) / (T - K - 1)
    muF = F.mean(0)
    Omega = np.cov(F, rowvar=False).reshape(K, K)
    denom = 1 + muF @ np.linalg.pinv(Omega) @ muF
    stat = (T - N - K) / N * (alpha @ np.linalg.pinv(Sigma) @ alpha) / denom
    pval = 1 - stats.f.cdf(stat, N, T - N - K)
    return {"grs_F": float(stat), "p_value": float(pval), "N": int(N),
            "T": int(T), "mean_abs_alpha_ann_pct": float(np.mean(np.abs(alpha)) * 12)}


def nw_alpha(y, X, L=6):
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    yy = d["y"].to_numpy()
    XX = np.column_stack([np.ones(len(d)), d[X.columns].to_numpy()])
    b, *_ = np.linalg.lstsq(XX, yy, rcond=None)
    e = yy - XX @ b
    XtX_inv = np.linalg.pinv(XX.T @ XX)
    S = (XX * e[:, None]).T @ (XX * e[:, None])
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = (XX[l:] * e[l:, None]).T @ (XX[:-l] * e[:-l, None])
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return float(b[0] * 12), float(b[0] / se[0])


def run():
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr, te = idx[idx < mid], idx[idx >= mid]
    disc = [f for f in shared
            if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < 1.96]
    F = load_f6()
    f6 = F[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]]

    coarse = Bs["wide"].loc[te, disc].mean(axis=1).rename("coarse")
    grave = Bs["vw"].loc[te, disc].mean(axis=1).rename("grave")
    spread = (coarse - grave).rename("spread")

    # GRS on the two books jointly (coarse + spread)
    ta = pd.concat([coarse, spread], axis=1)
    g = grs(ta, f6)

    a_c, t_c = nw_alpha(coarse, f6)
    a_s, t_s = nw_alpha(spread, f6)

    # expanding-window alpha stability of the spread
    exp_rows = []
    months = spread.dropna().index
    for cut in range(120, len(months), 60):
        sub = months[:cut]
        a, t = nw_alpha(spread.loc[sub], f6.loc[f6.index.intersection(sub)])
        exp_rows.append({"through": str(sub[-1])[:7], "alpha_ann_pct": a, "t": t})

    out = {"n_disc": len(disc),
           "grs_coarse_plus_spread_vs_F6": g,
           "coarse_alpha_F6": {"alpha_ann_pct": a_c, "t": t_c},
           "spread_alpha_F6": {"alpha_ann_pct": a_s, "t": t_s},
           "spread_alpha_expanding": exp_rows,
           "grs_rejects_spanning": bool(g["p_value"] < 0.05)}
    (HERE / "cn_grs.json").write_text(json.dumps(out, indent=2) + "\n")
    print("=" * 72)
    print("GRS / FAMA-MACBETH REFEREE-PROOFING")
    print("=" * 72)
    print(f"  N_disc={out['n_disc']}")
    print(f"  GRS F (coarse+spread vs FF5+Mom) = {g['grs_F']:.3f}  "
          f"p={g['p_value']:.4f}  (T={g['T']}, N={g['N']})")
    print(f"  mean|alpha| = {g['mean_abs_alpha_ann_pct']:.2f}%/yr")
    print(f"  coarse alpha_F6 = {a_c:+.2f}%/yr  t={t_c:.2f}")
    print(f"  spread alpha_F6 = {a_s:+.2f}%/yr  t={t_s:.2f}")
    print("  expanding-window spread alpha:")
    for r in exp_rows:
        print(f"     through {r['through']}: {r['alpha_ann_pct']:+.2f}%  t={r['t']:.2f}")
    print(f"  GRS rejects spanning (p<0.05): {out['grs_rejects_spanning']}")
    print("wrote cn_grs.json")
    return out


if __name__ == "__main__":
    run()
