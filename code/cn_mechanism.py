"""ECONOMIC MECHANISM of the coarse reallocation: what risk premium is it, and
how much capital can it hold. Everything here uses data on disk (OSAP monthly
factor returns, Ken French factors). Experiments that would need 13F / Markit /
IBES / CRSP-intraday (who-is-on-the-other-side, lending fees, analyst coverage,
intraday execution) are NOT faked; they are listed as data-gated in the report.

Blocks that run on available data:
  (A) Risk-premium decomposition. Regress the reallocation spread (coarse minus
      grave, pure venue value) on traded risk proxies BUILT FROM OSAP itself:
      a liquidity leg (Illiquidity), a short-term reversal leg (STreversal), an
      idiosyncratic-vol leg (IdioVol3F), a tail leg (BetaTailRisk), and a
      liquidity-beta leg (BetaLiquidityPS), on top of FF5+Mom. Report which
      priced factor absorbs the spread and the residual alpha.
  (B) Financial capacity. Square-root market-impact calibration of net Sharpe
      vs AUM for the coarse small-cap book, with a transparent ADV assumption,
      reported as a curve and a breakeven AUM.
  (C) Macro conditionality. The IP-growth state dependence already estimated in
      cn_state_route.json, re-expressed as a crowding/decay reading.

Writes cn_mechanism.json.
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

def _srs(x):
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
    return ff.join(mom, how="inner")


def nw_reg(y, X, L=6):
    """OLS of y on [1,X] with Newey-West t on every coef. Returns dict."""
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
    r2 = 1 - np.sum(e ** 2) / np.sum((yy - yy.mean()) ** 2)
    names = ["alpha"] + list(X.columns)
    return {"alpha_ann_pct": float(b[0] * 12), "t_alpha": float(b[0] / se[0]),
            "r2": float(r2), "n": int(len(d)),
            "betas": {nm: {"b": float(b[i]), "t": float(b[i] / se[i])}
                      for i, nm in enumerate(names)}}
def build_books(Bs, disc, te):
    grave = Bs["vw"].loc[te, disc].mean(axis=1).dropna()
    coarse = Bs["wide"].loc[te, disc].mean(axis=1).dropna()
    return grave, coarse, (coarse - grave).dropna()


def risk_proxies(Bs, te):
    """Traded risk-factor legs built from OSAP long-short factors themselves
    (percent units), aligned to the test window."""
    legs = {"Illiq": "Illiquidity", "STrev": "STreversal",
            "IdioVol": "IdioVol3F", "TailBeta": "BetaTailRisk",
            "LiqBeta": "BetaLiquidityPS"}
    out = {}
    for nm, f in legs.items():
        if f in Bs["wide"].columns:
            s = Bs["wide"].loc[te, f].dropna()
            if s.std() > 0:
                out[nm] = s
    return pd.DataFrame(out)


def risk_decomposition(spread, F, proxies):
    f6 = F[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]]
    res = {}
    res["F6"] = nw_reg(spread, f6)
    # add each proxy one at a time on top of F6, and all together
    for nm in proxies.columns:
        X = pd.concat([f6, proxies[[nm]]], axis=1)
        res["F6+" + nm] = nw_reg(spread, X)
    Xall = pd.concat([f6, proxies], axis=1)
    res["F6+all"] = nw_reg(spread, Xall)
    return res
def capacity_curve(coarse_book, adv_usd_daily=5e8, c_impact=0.5,
                   ann_turnover=2.0):
    """Square-root market-impact capacity curve for the coarse small-cap book.

    Net Sharpe at AUM A:
      SR_net(A) = SR_gross - (ann_cost(A) / vol_ann)
    where the per-rebalance one-way impact cost (fraction of traded notional) is
      cost_oneway(A) = c_impact * sigma_daily * sqrt( participation )
      participation  = (A * weight_turnover_per_rebal) / adv_usd_daily
    We use a transparent, conservative small-cap ADV aggregate `adv_usd_daily`
    (total daily dollar volume accessible to the book) and the betting-against
    small-cap impact coefficient c_impact=0.5 (Frazzini-Israel-Moskowitz is
    lower; 0.5 is deliberately conservative). ann_turnover is round-trips/yr.
    Returns the curve and the breakeven AUM where SR_net hits 1.0 and 0.5."""
    sr_gross = _srs(coarse_book)
    mu_m = coarse_book.mean() / 100.0           # monthly decimal
    vol_ann = coarse_book.std(ddof=1) / 100.0 * np.sqrt(12)
    sigma_daily = coarse_book.std(ddof=1) / 100.0 / np.sqrt(21)
    aums = [1e7, 5e7, 1e8, 5e8, 1e9, 5e9, 1e10]
    curve = []
    for A in aums:
        # notional traded per year = A * ann_turnover (one-way legs = 2x)
        part = (A * ann_turnover) / (adv_usd_daily * 252)  # frac of annual volume
        # per unit traded, sqrt impact on the daily scale
        cost_oneway = c_impact * sigma_daily * np.sqrt(max(part, 0.0))
        ann_cost = cost_oneway * ann_turnover * 2           # round trips, 2 legs
        sr_net = sr_gross - (ann_cost / vol_ann if vol_ann > 0 else 0)
        curve.append({"aum_usd": A, "participation": float(part),
                      "ann_cost_frac": float(ann_cost),
                      "sr_net": float(sr_net)})
    def breakeven(level):
        prev = None
        for r in curve:
            if r["sr_net"] < level:
                if prev is None:
                    return r["aum_usd"]
                # log-linear interp
                import math
                x0, y0 = math.log10(prev["aum_usd"]), prev["sr_net"]
                x1, y1 = math.log10(r["aum_usd"]), r["sr_net"]
                xi = x0 + (x1 - x0) * (y0 - level) / (y0 - y1)
                return float(10 ** xi)
            prev = r
        return None
    return {"sr_gross": sr_gross, "vol_ann": float(vol_ann),
            "adv_usd_daily": adv_usd_daily, "c_impact": c_impact,
            "ann_turnover": ann_turnover, "curve": curve,
            "aum_at_sr1": breakeven(1.0), "aum_at_sr05": breakeven(0.5)}


def cross_asset(F):
    """HKM cross-asset test: within each asset class (Commod, US bonds, FX, CDS,
    Options, Sovereign, FF25 equity), measure the DISPERSION portfolio = top-
    minus-bottom characteristic portfolio, and test whether its return loads on
    the intermediary-capital risk factor common across classes. This is the
    cross-asset analogue of the venue dispersion: a common risk premium behind
    the segment spread. Honest scope: HKM portfolios are characteristic sorts,
    not our venue sorts, so this tests the SHARED risk-premium hypothesis, not
    a literal relabeling of the equity reallocation."""
    hkm = pd.read_csv(Path(os.environ.get("LOCALSTRENGTH_DATA", "E:/BOAT-FX-DATA"))
                      / "hkm"
                      / "He_Kelly_Manela_Factors_And_Test_Assets_monthly.csv")
    hkm = hkm.dropna(subset=["yyyymm"])
    hkm["date"] = pd.to_datetime(hkm["yyyymm"].astype(float).astype(int)
                                 .astype(str), format="%Y%m")
    hkm = hkm.set_index("date")
    classes = {"Commod": "Commod", "USbond": "US_bonds", "FX": "FX",
               "CDS": "CDS", "Options": "Options", "Sov": "Sov_bonds",
               "Equity": "FF25"}
    icr = ("intermediary_capital_risk_factor"
           if "intermediary_capital_risk_factor" in hkm.columns else None)
    out = {}
    for nm, pref in classes.items():
        cols = [c for c in hkm.columns if c.startswith(pref)]
        if len(cols) < 3:
            continue
        sub = hkm[cols].apply(pd.to_numeric, errors="coerce")
        disp = (sub[cols[-1]] - sub[cols[0]]).dropna()   # top-minus-bottom
        rec = {"n_portfolios": len(cols), "disp_mean_ann_pct": float(disp.mean() * 12),
               "disp_sharpe": _srs(disp)}
        if icr:
            d = pd.concat([disp.rename("y"),
                           pd.to_numeric(hkm[icr], errors="coerce").rename("icr")],
                          axis=1).dropna()
            if len(d) > 36 and d["icr"].std() > 0:
                b = np.polyfit(d["icr"], d["y"], 1)
                corr = float(np.corrcoef(d["icr"], d["y"])[0, 1])
                rec["beta_intermediary"] = float(b[0])
                rec["corr_intermediary"] = corr
        out[nm] = rec
    return out
def run():
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr, te = idx[idx < mid], idx[idx >= mid]
    disc = [f for f in shared
            if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < 1.96]
    F = load_f6()
    grave, coarse, spread = build_books(Bs, disc, te)
    proxies = risk_proxies(Bs, te)

    risk = risk_decomposition(spread, F, proxies)
    cap = capacity_curve(coarse)
    xa = cross_asset(F)

    out = {
        "n_disc": len(disc),
        "risk_decomposition": risk,
        "capacity": cap,
        "cross_asset_hkm": xa,
        "data_gated_experiments": [
            "who-is-on-the-other-side (needs 13F institutional holdings)",
            "lending fees / short interest (needs Markit / Compustat short)",
            "analyst-coverage mediation (needs IBES)",
            "intraday execution alpha (needs TAQ)",
        ],
    }
    (HERE / "cn_mechanism.json").write_text(json.dumps(out, indent=2) + "\n")
    _print(out)
    return out


def _print(o):
    print("=" * 78)
    print("ECONOMIC MECHANISM: risk premium, capacity, cross-asset")
    print("=" * 78)
    r = o["risk_decomposition"]
    print("  (A) RISK-PREMIUM DECOMPOSITION of reallocation spread:")
    print(f"      {'model':12s} {'alpha%':>7s} {'t':>6s} {'R2':>5s}")
    for k, v in r.items():
        print(f"      {k:12s} {v['alpha_ann_pct']:7.2f} {v['t_alpha']:6.2f} "
              f"{v['r2']:5.2f}")
    print("      significant proxy loadings (on top of F6, |t|>2):")
    for k, v in r.items():
        if k.startswith("F6+") and k != "F6+all":
            leg = k[3:]
            bt = v["betas"].get(leg)
            if bt and abs(bt["t"]) > 2:
                print(f"        {leg}: beta={bt['b']:+.3f} t={bt['t']:+.2f} "
                      f"-> alpha falls to {v['alpha_ann_pct']:.2f}%")
    c = o["capacity"]
    print("-" * 78)
    print(f"  (B) FINANCIAL CAPACITY (sqrt impact, c={c['c_impact']}, "
          f"ADV=${c['adv_usd_daily']:.0e}/day, gross SR={c['sr_gross']:.2f}):")
    for row in c["curve"]:
        print(f"      AUM ${row['aum_usd']:>10.0e}: particip="
              f"{row['participation']*100:6.2f}%  net SR={row['sr_net']:+.3f}")
    print(f"      breakeven: SR=1.0 at ${c['aum_at_sr1']:.2e}, "
          f"SR=0.5 at ${c['aum_at_sr05']:.2e}" if c['aum_at_sr1'] else
          f"      SR stays >1 across range")
    print("-" * 78)
    print("  (C) CROSS-ASSET (HKM, dispersion portfolio & intermediary beta):")
    print(f"      {'class':8s} {'nport':>5s} {'disp SR':>8s} {'interm beta':>11s} "
          f"{'corr':>6s}")
    for nm, v in o["cross_asset_hkm"].items():
        print(f"      {nm:8s} {v['n_portfolios']:5d} {v.get('disp_sharpe',float('nan')):8.2f} "
              f"{v.get('beta_intermediary',float('nan')):11.3f} "
              f"{v.get('corr_intermediary',float('nan')):6.2f}")
    print("-" * 78)
    print("  DATA-GATED (not run; require firm-level data not on disk):")
    for s in o["data_gated_experiments"]:
        print("     -", s)
    print("wrote cn_mechanism.json")


if __name__ == "__main__":
    run()
