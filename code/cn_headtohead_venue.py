"""
cn_headtohead.py
================
Head-to-head vs top-fund PROXY strategies (plan: head-to-head-funds).

Standard-practice weapons a top quant shop deploys on the SAME discarded factors:
  P1  universal equal-weight (vw)            -- naive baseline
  P2  risk-parity on the discards (wide)     -- "risk weighting" desk
  P3  FF6+BAB canon mimicking portfolio      -- factor-investing desk
  P4  hierarchical risk-parity (best weighting from cn_weighting.json)

The salvage/coarse book (wide EW reallocation) is tested for INCREMENTAL alpha
against each proxy via a spanning regression (salvage ~ a + b*proxy + controls).
If alpha > 0 with t > 2 AFTER the proxy + FF6 + BAB, the resolution move adds
something no standard weighting desk captures. We also report the reverse: does
the proxy span the salvage book?

Honest framing: fancy weighting (P2/P4) can post a HIGHER raw Sharpe, but prior
results show it is largely a low-beta/BAB relabel. The clean claim is ORTHOGONAL
alpha, not raw Sharpe.

Writes cn_headtohead.json.
"""
import json, os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from cn_resurrection import load_battlefields, _tstat, BATTLES

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
KF = Path("E:/BOAT-FX-DATA/ken_french")
OUT = HERE / "cn_headtohead.json"
ANN = np.sqrt(12.0)


def sr(x):
    x = pd.Series(x).dropna()
    return float(ANN * x.mean() / x.std(ddof=1)) if x.size >= 24 and x.std(ddof=1) else np.nan


def load_f6():
    ff = pd.read_csv(KF / "FF5_monthly.csv")
    ff["Date"] = pd.to_datetime(ff["Date"].astype(str), format="%Y%m")
    ff = ff.set_index("Date"); ff.index = ff.index.to_period("M").to_timestamp("M")
    mom = pd.read_csv(KF / "Momentum_monthly.csv")
    mom["Date"] = pd.to_datetime(mom["Date"].astype(str), format="%Y%m")
    mom = mom.set_index("Date"); mom.index = mom.index.to_period("M").to_timestamp("M")
    return ff.join(mom, how="inner")


def ols_alpha(y, X):
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(d) < 48:
        return None
    yy = d["y"].to_numpy()
    XX = np.column_stack([np.ones(len(d)), d[X.columns].to_numpy()])
    beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
    resid = yy - XX @ beta
    XtX_inv = np.linalg.pinv(XX.T @ XX)
    Sx = (XX * resid[:, None]).T @ (XX * resid[:, None])
    for l in range(1, 13):
        w = 1 - l / 13
        G = (XX[l:] * resid[l:, None]).T @ (XX[:-l] * resid[:-l, None])
        Sx += w * (G + G.T)
    cov = XtX_inv @ Sx @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return {"alpha_ann_pct": float(beta[0] * 12), "t": float(beta[0] / se[0]),
            "n": int(len(d))}


def run():
    Bs, shared, idx = load_battlefields()
    idx = pd.DatetimeIndex(idx); mid = idx[len(idx) // 2]
    tr, te = idx[idx < mid], idx[idx >= mid]
    disc = [f for f in shared
            if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < 1.96]
    F = load_f6()

    # proxy strategies on the discarded factors, test window
    Rw = Bs["wide"].loc[te, disc]
    Rvw = Bs["vw"].loc[te, disc]
    salvage = Rw.mean(axis=1).dropna()                    # coarse reallocation (hero)
    p1_univ = Rvw.mean(axis=1).dropna()                   # naive universal
    # risk parity (inverse-vol weights estimated on train)
    vol_tr = Bs["wide"].loc[tr, disc].std()
    w_rp = (1.0 / vol_tr) / (1.0 / vol_tr).sum()
    p2_rp = (Rw * w_rp).sum(axis=1).dropna()
    # BAB from discards (low-vol minus high-vol, in wide)
    order = vol_tr.sort_values().index
    lo, hi = order[:len(order)//3], order[-len(order)//3:]
    bab = (Bs["wide"].loc[te, lo].mean(axis=1) - Bs["wide"].loc[te, hi].mean(axis=1)).rename("BAB")

    mkt = F[["Mkt-RF"]]
    f6 = F[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]]
    f6b = pd.concat([f6, bab], axis=1)

    out = {"n_disc": len(disc), "test": [str(te.min())[:7], str(te.max())[:7]],
           "raw_sharpe": {"salvage_coarse": sr(salvage), "P1_universal": sr(p1_univ),
                          "P2_risk_parity": sr(p2_rp)},
           "spanning_salvage_increment": {}, "reverse_proxy_increment": {}}

    # does salvage earn alpha beyond each proxy (+ FF6 + BAB)?
    for name, proxy in [("vs_universal", p1_univ), ("vs_risk_parity", p2_rp)]:
        X = pd.concat([proxy.rename("PROXY"), f6b], axis=1)
        out["spanning_salvage_increment"][name] = ols_alpha(salvage, X)
    # pure canon only (no proxy book), the factor-investing desk
    out["spanning_salvage_increment"]["vs_FF6_BAB_only"] = ols_alpha(salvage, f6b)

    # reverse: does the risk-parity book earn alpha beyond salvage + FF6 + BAB?
    Xrev = pd.concat([salvage.rename("SALVAGE"), f6b], axis=1)
    out["reverse_proxy_increment"]["risk_parity_vs_salvage"] = ols_alpha(p2_rp, Xrev)
    out["reverse_proxy_increment"]["universal_vs_salvage"] = ols_alpha(p1_univ, Xrev)

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("=" * 70)
    print("HEAD-TO-HEAD: salvage (coarse reallocation) vs top-fund proxies")
    print(f"  N_disc={out['n_disc']} test={out['test']}")
    print("  raw Sharpe:", {k: round(v, 3) for k, v in out["raw_sharpe"].items()})
    print("  salvage incremental alpha (ann%, HAC t):")
    for k, v in out["spanning_salvage_increment"].items():
        if v: print(f"     {k:22s} alpha={v['alpha_ann_pct']:+6.2f}% t={v['t']:5.2f}")
    print("  reverse (proxy alpha beyond salvage):")
    for k, v in out["reverse_proxy_increment"].items():
        if v: print(f"     {k:26s} alpha={v['alpha_ann_pct']:+6.2f}% t={v['t']:5.2f}")
    print("wrote", OUT)
    return out


if __name__ == "__main__":
    run()
