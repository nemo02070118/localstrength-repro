"""
cn_ashare_rigor.py
==================
Rigor battery for the A-share salvage result (plan: ashare-rigor).

Four referee-proof tests, all long-only, net of cost, on the tradable universe:

  (R1) SPANNING / ORTHOGONALITY -- does the salvage book earn alpha against
       (a) the A-share equal-weight market, and (b) the UNIVERSAL-venue versions
       of the same factors ("standard practice canon")?  If alpha survives, the
       resolution gain is NOT just a repackaging of the same factor premia.

  (R2) RANDOM-VENUE PLACEBO -- randomly reassign stocks to venues 1000x; the
       coarse-focusing gain should vanish.  True gain vs placebo distribution.

  (R3) FDR across factors -- Benjamini-Hochberg on the per-factor routing gains,
       so "venue helps" is not a multiple-testing artifact.

  (R4) NEWEY-WEST t-stats on every book's mean monthly excess return.

Reuses the panel + estimators from cn_ashare_salvage (its heavy code is
__main__-guarded, so importing is cheap-ish but still loads the h5).

Writes cn_ashare_rigor.json.
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

import cn_ashare_salvage as S   # load_monthly, build_monthly, ls_return, venue_masks, sharpe

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "cn_ashare_rigor.json"
RNG = np.random.default_rng(7)

# ---- globals shared with parallel placebo workers (set via initializer) ----
_W = {}

def _pl_init(Rfin, trad_np, SIGfin_list, facs_ok, tr_mask, te_mask, univ_sharpe, seed0):
    _W["Rfin"] = Rfin
    _W["trad"] = trad_np
    _W["SIG"] = SIGfin_list          # list of (Ninst-aligned) arrays
    _W["facs"] = facs_ok
    _W["tr"] = tr_mask
    _W["te"] = te_mask
    _W["usr"] = univ_sharpe
    _W["seed0"] = seed0

def _sharpe_np(x, ann=12):
    x = x[np.isfinite(x)]
    if len(x) < 6 or x.std() == 0:
        return np.nan
    return x.mean() / x.std() * np.sqrt(ann)

def _book(labels, g):
    Rfin = _W["Rfin"]; trad = _W["trad"]; T = Rfin.shape[0]
    in_grp = (labels[None, :] == g) & trad
    acc = np.zeros(T); cnt = np.zeros(T)
    for sgf in _W["SIG"]:
        sg = np.where(in_grp, sgf, np.nan)
        feas = np.sum(np.isfinite(sg) & np.isfinite(Rfin), axis=1)
        thr = np.nanquantile(np.where(np.isfinite(Rfin), sg, np.nan), 0.8, axis=1)
        longs = sg >= thr[:, None]
        rr = np.where(np.isfinite(sg) & np.isfinite(Rfin), Rfin, np.nan)
        nlong = np.sum(longs & np.isfinite(rr), axis=1)
        long_ret = np.nansum(np.where(longs, rr, 0.0), axis=1) / np.maximum(nlong, 1)
        fac_ret = long_ret - np.nanmean(rr, axis=1)
        good = (feas >= 20) & np.isfinite(fac_ret) & (nlong > 0)
        acc[good] += fac_ret[good]; cnt[good] += 1
    book = np.full(T, np.nan); book[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0]
    return book

def _pl_draw(b):
    rng = np.random.default_rng(_W["seed0"] + b)
    Ninst = _W["Rfin"].shape[1]
    lab = rng.integers(0, 3, size=Ninst)
    tr, te = _W["tr"], _W["te"]
    tr_sh, books_te = [], []
    for g in range(3):
        bk = _book(lab, g)
        tr_sh.append(_sharpe_np(bk[tr])); books_te.append(bk[te])
    gbest = int(np.nanargmax([s if np.isfinite(s) else -9 for s in tr_sh]))
    return _sharpe_np(books_te[gbest]) - _W["usr"]


def nw_tstat(x, lags=6):
    """Newey-West t-stat for the mean of a monthly series."""
    x = pd.Series(x).dropna().values
    n = len(x)
    if n < 12:
        return (np.nan, np.nan)
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        cov = (e[L:] @ e[:-L]) / n
        var += 2 * w * cov
    se = np.sqrt(var / n)
    return (float(mu), float(mu / se)) if se > 0 else (float(mu), np.nan)


def ols_alpha(y, regressors, lags=6):
    """Regress y on [1, regressors]; return intercept (alpha), its NW t, betas.
    regressors: a list of Series and/or DataFrames."""
    parts = [y]
    for r in regressors:
        parts.append(r)
    df = pd.concat(parts, axis=1).dropna()
    if len(df) < 24:
        return None
    yy = df.iloc[:, 0].values
    XX = np.column_stack([np.ones(len(df))] + [df.iloc[:, j].values
                                               for j in range(1, df.shape[1])])
    beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
    resid = yy - XX @ beta
    n, k = XX.shape
    XtX_inv = np.linalg.inv(XX.T @ XX)
    # Newey-West HAC covariance
    S_hac = (resid[:, None] * XX).T @ (resid[:, None] * XX) / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        u = resid[:, None] * XX
        G = u[L:].T @ u[:-L] / n
        S_hac += w * (G + G.T)
    cov = XtX_inv @ (n * S_hac) @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return {"alpha_monthly": float(beta[0]),
            "alpha_ann_pct": float(beta[0] * 12 * 100),
            "t_alpha": float(beta[0] / se[0]) if se[0] > 0 else np.nan,
            "betas": [float(b) for b in beta[1:]],
            "n": int(n)}


if __name__ == "__main__":
    print("[1] load + monthly panel ...")
    d = S.load_monthly()
    m = S.build_monthly(d)
    del d
    R = m["ret_m"].unstack("inst")
    DV = m["dvol"].unstack("inst")
    VOLd = m["vol_d"].unstack("inst").reindex_like(R)
    MAXd = m["max_d"].unstack("inst").reindex_like(R)
    AMI = m["amihud"].unstack("inst").reindex_like(R)
    R = R.sort_index(); DV = DV.sort_index()
    SIG = {
        "REV":  -R,
        "MOM":   R.rolling(12).sum().shift(1),
        "VOL":  -VOLd,
        "MAX":  -MAXd,
        "TURN": -DV.pct_change(),
        "ILLIQ": AMI,
    }
    facs_ok = [f for f in SIG if f != "ILLIQ"]
    sv_venues = ["small", "mid", "large"]
    FLOOR, TCOST = 0.30, 0.001
    print("[2] build venue series + train/test split ...")
    liq_rank = DV.rank(axis=1, pct=True)
    tradable = liq_rank > FLOOR
    Rt = R.where(tradable)
    VEN = S.venue_masks(DV.where(tradable))
    venues = sv_venues + ["all"]
    series = {}
    for fac, sig in SIG.items():
        for ven in venues:
            vm = None if ven == "all" else VEN[ven]
            series[(fac, ven)] = S.ls_return(sig, Rt, vm, tcost=TCOST, long_only=True)
    dates = R.index
    split = len(dates) // 2
    tr_idx, te_idx = dates[:split], dates[split:]

    def win(key, idx):
        return pd.Series(series[key]).reindex(idx)

    def combo(parts):
        return pd.concat(parts, axis=1).mean(axis=1)

    # per-factor train-best venue (fine) and whole-book coarse venue
    route = {}
    for fac in SIG:
        tr = {v: S.sharpe(win((fac, v), tr_idx)) for v in sv_venues}
        route[fac] = max(sv_venues, key=lambda v: tr[v] if np.isfinite(tr[v]) else -9)
    coarse_tr = {v: S.sharpe(combo([win((f, v), tr_idx) for f in facs_ok]))
                 for v in sv_venues}
    coarse_venue = max(sv_venues, key=lambda v: coarse_tr[v] if np.isfinite(coarse_tr[v]) else -9)
    print("    coarse venue:", coarse_venue, "| per-factor routes:", route)

    # the three books on the TEST period (ex-ILLIQ)
    book_universal = combo([win((f, "all"), te_idx) for f in facs_ok])
    book_coarse    = combo([win((f, coarse_venue), te_idx) for f in facs_ok])
    book_fine      = combo([win((f, route[f]), te_idx) for f in facs_ok])
    print("[3] R1 spanning: salvage alpha vs market + universal-venue canon ...")
    # market = EW return of tradable universe (excess over cross-sec mean is ~0,
    # so use the raw EW market return as the systematic factor)
    mkt = Rt.mean(axis=1).reindex(te_idx)
    # universal-venue factor returns (standard practice "canon")
    canon = {f: win((f, "all"), te_idx) for f in facs_ok}
    Xcanon = pd.concat([mkt.rename("MKT")] + [v.rename(f"U_{k}") for k, v in canon.items()],
                       axis=1)
    span = {
        "fine_vs_mkt":        ols_alpha(book_fine, [mkt.rename("MKT")]),
        "fine_vs_mkt_canon":  ols_alpha(book_fine, [Xcanon]),
        "coarse_vs_mkt":      ols_alpha(book_coarse, [mkt.rename("MKT")]),
        "coarse_vs_mkt_canon":ols_alpha(book_coarse, [Xcanon]),
    }
    for k, v in span.items():
        if v: print(f"    {k}: alpha={v['alpha_ann_pct']:.2f}%/yr  t={v['t_alpha']:.2f}")

    N_PLACEBO = 1000
    print(f"[4] R2 random-venue placebo ({N_PLACEBO} draws, vectorized) ...")
    # true coarse gain over universal on TEST
    true_gain = S.sharpe(book_coarse) - S.sharpe(book_universal)

    # ---- vectorized machinery -----------------------------------------------
    # Build numpy tensors aligned on (month T x stock Ninst). For each factor,
    # the lagged signal S_f[t,i] (known at t, predicting R[t+1]) and realized
    # next-month return. A long-only venue book = mean next-month return of the
    # top-quintile-by-signal names *within the venue* minus the venue EW mean.
    cols = Rt.columns
    Ridx = Rt.index
    Rnp = Rt.values.astype(float)                 # realized monthly return
    trad_np = tradable.reindex_like(Rt).values     # bool tradable each month
    T, Ninst = Rnp.shape
    # signal tensors (shift by 1 month to be known at prior month-end)
    SIGnp = {f: SIG[f].reindex_like(Rt).shift(1).values.astype(float) for f in facs_ok}
    te_mask = np.array([d in set(te_idx) for d in Ridx])
    tr_mask = np.array([d in set(tr_idx) for d in Ridx])

    Rfin = np.where(np.isfinite(Rnp), Rnp, np.nan)
    SIGfin = {f: np.where(np.isfinite(SIGnp[f]), SIGnp[f], np.nan) for f in facs_ok}

    def book_from_labels(labels, group_sel):
        """Vectorized over months. labels:(Ninst,) group id per stock.
        Returns monthly book series (len T) = EW across facs_ok of
        (top-quintile long - venue EW) within group_sel venue."""
        in_grp = (labels[None, :] == group_sel) & trad_np   # (T,Ninst) bool
        acc = np.zeros(T); cnt = np.zeros(T)
        for f in facs_ok:
            sg = np.where(in_grp, SIGfin[f], np.nan)        # masked signal
            # per-month feasibility: need >=20 names with finite signal & return
            feas_cnt = np.sum(np.isfinite(sg) & np.isfinite(Rfin), axis=1)
            thr = np.nanquantile(np.where(np.isfinite(Rfin), sg, np.nan),
                                 0.8, axis=1)               # (T,)
            longs = sg >= thr[:, None]                      # (T,Ninst)
            rr = np.where(np.isfinite(sg) & np.isfinite(Rfin), Rfin, np.nan)
            long_ret = np.nansum(np.where(longs, rr, 0.0), axis=1) / \
                       np.maximum(np.sum(longs & np.isfinite(rr), axis=1), 1)
            bench_ret = np.nanmean(rr, axis=1)
            fac_ret = long_ret - bench_ret
            good = (feas_cnt >= 20) & np.isfinite(fac_ret) & \
                   (np.sum(longs & np.isfinite(rr), axis=1) > 0)
            acc[good] += fac_ret[good]; cnt[good] += 1
        book = np.full(T, np.nan)
        book[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0]
        return book

    def sharpe_np(x, ann=12):
        x = x[np.isfinite(x)]
        if len(x) < 6 or x.std() == 0:
            return np.nan
        return x.mean() / x.std() * np.sqrt(ann)

    from concurrent.futures import ProcessPoolExecutor
    import os
    SIGfin_list = [SIGfin[f] for f in facs_ok]
    univ_sharpe = float(S.sharpe(book_universal))
    nproc = min(20, os.cpu_count() or 4)
    print(f"    dispatching {N_PLACEBO} draws across {nproc} cores ...")
    with ProcessPoolExecutor(max_workers=nproc, initializer=_pl_init,
                             initargs=(Rfin, trad_np, SIGfin_list, facs_ok,
                                       tr_mask, te_mask, univ_sharpe, 10_000)) as ex:
        placebo = list(ex.map(_pl_draw, range(N_PLACEBO), chunksize=10))
    placebo = np.array([p for p in placebo if np.isfinite(p)])
    plc = {
        "true_coarse_gain": float(true_gain),
        "placebo_mean": float(placebo.mean()),
        "placebo_sd": float(placebo.std()),
        "placebo_p95": float(np.percentile(placebo, 95)),
        "emp_p_true_ge": float((placebo >= true_gain).mean()),
        "z_true_vs_placebo": float((true_gain - placebo.mean()) / placebo.std()),
    }
    print(f"    true gain {true_gain:.3f} vs placebo mean {placebo.mean():.3f} "
          f"(z={plc['z_true_vs_placebo']:.2f}, p={plc['emp_p_true_ge']:.3f})")
    print("[5] R3 FDR across factors: does routing beat universal per factor? ...")
    # per-factor: diff series (routed venue - universal) on TEST; NW t and p
    from math import erfc, sqrt
    def p_from_t(t, n):
        # two-sided normal approx
        return erfc(abs(t) / sqrt(2))
    perfac = {}
    pvals = []
    for f in facs_ok:
        diff = (win((f, route[f]), te_idx) - win((f, "all"), te_idx))
        mu, t = nw_tstat(diff)
        p = p_from_t(t, len(te_idx)) if np.isfinite(t) else 1.0
        perfac[f] = {"route_to": route[f], "mean_diff_monthly": mu, "t": t, "p": p}
        pvals.append((f, p))
    # Benjamini-Hochberg
    pvals_sorted = sorted(pvals, key=lambda x: x[1])
    mtests = len(pvals_sorted)
    bh = {}
    for rank, (f, p) in enumerate(pvals_sorted, start=1):
        crit = 0.05 * rank / mtests
        bh[f] = {"p": p, "bh_crit_0.05": crit, "pass": bool(p <= crit)}
    n_pass = sum(1 for v in bh.values() if v["pass"])
    print(f"    {n_pass}/{mtests} factors pass BH q=0.05 for routing>universal")

    print("[6] R4 Newey-West t on each book mean excess ...")
    nw = {
        "universal": nw_tstat(book_universal),
        "coarse":    nw_tstat(book_coarse),
        "fine":      nw_tstat(book_fine),
    }
    for k, (mu, t) in nw.items():
        print(f"    {k}: mean={mu*12*100:.2f}%/yr  NW-t={t:.2f}")

    print("[7] writing JSON ...")
    def clean(o):
        if isinstance(o, dict):  return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(x) for x in o]
        if isinstance(o, float): return None if not np.isfinite(o) else round(o, 4)
        return o
    result = {
        "market": "China A-share (QuantaAlpha daily_pv.h5)",
        "spec": "long-only, floor=0.30, cost=10bps, ex-ILLIQ, OOS test half",
        "coarse_venue": coarse_venue,
        "per_factor_routes": route,
        "R1_spanning": clean(span),
        "R2_placebo": clean(plc),
        "R3_fdr": {"per_factor": clean(perfac), "bh_q0.05": clean(bh), "n_pass": n_pass,
                   "n_tests": mtests},
        "R4_newey_west": {k: {"mean_ann_pct": round(mu*12*100, 3), "t": round(t, 3)}
                          for k, (mu, t) in nw.items()},
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("    wrote", OUT)
