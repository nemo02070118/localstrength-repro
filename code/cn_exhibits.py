"""New exhibits for the JF-flagship paper. Produces, from the OSAP venue panels
and the Fama-French / q5 factors, the referee-demanded objects the current
draft lacks:

  (1) SUMMARY STATISTICS: per-venue cross-sectional distribution of factor
      Sharpe ratios, volatilities, and pairwise strength correlations across
      the four battlefields; plus the discard/keep split counts.
  (2) GRS / joint-alpha test: are the discarded factors' reallocation alphas
      jointly zero? (GRS 1989 F-test on the family-level reallocation alphas.)
  (3) CER / utility gain: certainty-equivalent return gain of the coarse book
      over the grave book for a mean-variance investor at gamma in {1,3,5,10}.
  (4) FAMILY DECOMPOSITION: per economic family, the discard count, its native
      battlefield, and the OOS grave vs coarse Sharpe.

Writes cn_exhibits.json. Decimal-return convention (OSAP percent / 100).
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


def _classify(f):
    """Economic family of an OSAP (Chen-Zimmermann) anomaly, by signal name."""
    s = f.lower()
    if any(k in s for k in ("mom", "high52", "indmom", "revenuesurprise",
            "earnsup", "numearn", "earningsconsistency", "season")):
        return "momentum"
    if any(k in s for k in ("reversal", "meanrank")):
        return "reversal"
    if any(k in s for k in ("bm", "ep", "sp", "cfp", "cf", "am", "entmult",
            "netdebtprice", "payout", "intanep", "intancfp",
            "equityduration", "price", "leverage", "oplever")):
        return "value"
    if any(k in s for k in ("assetgrowth", "investment", "grcapx", "grltnoa",
            "deleq", "delcol", "delfinl", "dellti", "delnetfin", "chnncoa",
            "cheq", "compequiss", "compositedebt", "dnoa", "hire", "gradexp",
            "adexp", "brandinvest")):
        return "investment"
    if any(k in s for k in ("operprof", "cboperprof", "roe", "grsale",
            "cashprod", "varcf", "rdability", "rd", "tax", "chtax")):
        return "profitability"
    if any(k in s for k in ("idiovol", "realizedvol", "volsd", "volmkt",
            "maxret", "beta", "coskew", "skew", "tailrisk", "returnskew")):
        return "risk"
    if any(k in s for k in ("illiquidity", "dolvol", "bidask", "zerotrade",
            "std_turn", "volumetrend", "liquidity", "pricedelay", "frontier")):
        return "liquidity"
    if any(k in s for k in ("size", "firmage", "herf", "tang")):
        return "size"
    return "other"


warnings.filterwarnings("ignore")
HERE = Path(os.environ.get("LOCALSTRENGTH_RESULTS", str(Path(__file__).resolve().parent.parent / "results")))
DATA = Path(os.environ.get("LOCALSTRENGTH_DATA", "E:/BOAT-FX-DATA"))
ANN = np.sqrt(12.0)


def _setup():
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr = idx[idx < mid]; te = idx[idx >= mid]
    disc = [f for f in shared if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < 1.96]
    keep = [f for f in shared if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) >= 1.96]
    pool = {n: np.nanmean([_sr(Bs[n].loc[tr, f]) for f in disc]) for n in BATTLES}
    cb = max(pool, key=pool.get)
    return Bs, shared, idx, tr, te, disc, keep, cb


def summary_stats(Bs, shared, tr, te, disc, keep):
    """Cross-sectional Sharpe distribution per venue (full sample), venue
    strength-vector correlations, and the discard/keep breakdown."""
    full = Bs[BATTLES[0]].index
    perv = {}
    S = {}
    for n in BATTLES:
        srs = np.array([_sr(Bs[n].loc[full, f]) for f in shared], float)
        srs = srs[np.isfinite(srs)]
        S[n] = pd.Series({f: _sr(Bs[n].loc[full, f]) for f in shared})
        perv[n] = {"n_factors": int(srs.size), "sr_mean": float(np.mean(srs)),
                   "sr_median": float(np.median(srs)), "sr_sd": float(np.std(srs)),
                   "sr_p10": float(np.percentile(srs, 10)),
                   "sr_p90": float(np.percentile(srs, 90))}
    SS = pd.DataFrame(S)
    corr = SS.corr().round(3).to_dict()
    return {"per_venue": perv, "venue_strength_corr": corr,
            "n_shared": len(shared), "n_discarded": len(disc),
            "n_kept": len(keep), "sample": [str(full.min())[:7], str(full.max())[:7]]}


def family_decomp(Bs, tr, te, disc, cb):
    """Per economic family: discard count, native battlefield (train-best for
    the family pool), OOS grave vs coarse Sharpe, and the reallocation gain."""
    from collections import defaultdict
    G = defaultdict(list)
    for f in disc:
        G[_classify(f)].append(f)
    out = {}
    for fam, fs in sorted(G.items(), key=lambda x: -len(x[1])):
        if len(fs) < 3:
            continue
        ft = {n: np.nanmean([_sr(Bs[n].loc[tr, f]) for f in fs]) for n in BATTLES}
        fb = max(ft, key=ft.get)
        _, mg = _book(Bs, fs, {f: "vw" for f in fs}, te)
        _, mc = _book(Bs, fs, {f: fb for f in fs}, te)
        out[fam] = {"n": len(fs), "native_bf": fb,
                    "sr_grave": _sr(mg / 100.0), "sr_coarse": _sr(mc / 100.0),
                    "gain": _sr(mc / 100.0) - _sr(mg / 100.0)}
    return out


def grs_test(Bs, tr, te, disc, cb):
    """GRS (1989) joint test that the family-level reallocation gains are jointly
    zero. Build the per-family gain return stream (coarse minus grave), stack
    into an N_fam x T matrix, and compute the GRS F-statistic against a zero
    benchmark (no factors; tests joint mean = 0)."""
    from collections import defaultdict
    from scipy import stats
    G = defaultdict(list)
    for f in disc:
        G[_classify(f)].append(f)
    streams = []
    for fam, fs in G.items():
        if len(fs) < 3:
            continue
        ft = {n: np.nanmean([_sr(Bs[n].loc[tr, f]) for f in fs]) for n in BATTLES}
        fb = max(ft, key=ft.get)
        _, mg = _book(Bs, fs, {f: "vw" for f in fs}, te)
        _, mc = _book(Bs, fs, {f: fb for f in fs}, te)
        streams.append(((mc - mg) / 100.0).rename(fam))
    R = pd.concat(streams, axis=1).dropna()
    T, Nf = R.shape
    mu = R.mean().values
    Sig = R.cov().values
    # GRS with no factors: F = (T-N)/N * mu' Sigma^-1 mu  ~ F(N, T-N)
    Sinv = np.linalg.pinv(Sig)
    W = float(mu @ Sinv @ mu)
    Fstat = (T - Nf) / Nf * W
    pval = float(1 - stats.f.cdf(Fstat, Nf, T - Nf))
    return {"n_families": int(Nf), "T": int(T), "grs_F": float(Fstat),
            "p_value": pval, "mean_gain_ann_pct": float(np.mean(mu) * 12 * 100)}


def cer_gain(Bs, tr, te, disc, cb):
    """Certainty-equivalent return gain of the coarse book over the grave book
    for a mean-variance investor, CER = mu - gamma/2 * var (annualized),
    reported for gamma in {1,3,5,10}."""
    _, mg = _book(Bs, disc, {f: "vw" for f in disc}, te)
    _, mc = _book(Bs, disc, {f: cb for f in disc}, te)
    mg = (mg / 100.0).dropna(); mc = (mc / 100.0).dropna()
    out = {}
    for g in (1, 3, 5, 10):
        cer_g = (mg.mean() - g / 2 * mg.var()) * 12
        cer_c = (mc.mean() - g / 2 * mc.var()) * 12
        out[str(g)] = {"cer_grave_pct": float(cer_g * 100),
                       "cer_coarse_pct": float(cer_c * 100),
                       "cer_gain_pct": float((cer_c - cer_g) * 100)}
    return out


def run():
    Bs, shared, idx, tr, te, disc, keep, cb = _setup()
    out = {
        "summary_stats": summary_stats(Bs, shared, tr, te, disc, keep),
        "family_decomp": family_decomp(Bs, tr, te, disc, cb),
        "grs_test": grs_test(Bs, tr, te, disc, cb),
        "cer_gain": cer_gain(Bs, tr, te, disc, cb),
        "coarse_bf": cb,
    }
    (HERE / "cn_exhibits.json").write_text(json.dumps(out, indent=2) + "\n")
    s = out["summary_stats"]
    print("=" * 74)
    print("NEW EXHIBITS")
    print("=" * 74)
    print(f"shared={s['n_shared']} discarded={s['n_discarded']} kept={s['n_kept']} "
          f"sample {s['sample'][0]}..{s['sample'][1]}")
    print("-- per-venue median factor Sharpe --")
    for n, v in s["per_venue"].items():
        print(f"   {n:5s}: median={v['sr_median']:+.3f} mean={v['sr_mean']:+.3f} "
              f"sd={v['sr_sd']:.3f}")
    print("-- family decomposition (OOS) --")
    for fam, v in out["family_decomp"].items():
        print(f"   {fam:13s} n={v['n']:>2} native={v['native_bf']:5s} "
              f"grave={v['sr_grave']:+.3f} coarse={v['sr_coarse']:+.3f} "
              f"gain={v['gain']:+.3f}")
    g = out["grs_test"]
    print(f"-- GRS joint test: F({g['n_families']},{g['T']})={g['grs_F']:.2f} "
          f"p={g['p_value']:.4f}  mean gain={g['mean_gain_ann_pct']:+.2f}%/yr")
    print("-- CER gain (coarse - grave) --")
    for gm, v in out["cer_gain"].items():
        print(f"   gamma={gm:>2}: {v['cer_gain_pct']:+.2f}%/yr")
    print("wrote cn_exhibits.json")
    return out


if __name__ == "__main__":
    run()

