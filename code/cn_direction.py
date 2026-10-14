"""DIRECTION vs MAGNITUDE  --  the untested axis.

Every prior experiment estimated the venue advantage MAGNITUDE (a continuous,
noisy quantity) and found it does not persist (phi=0.135). But routing only
needs the DIRECTION: which venue is a factor's home, not by how much. Direction
is a discrete/sign quantity and can be far more robust than magnitude -- like a
compass needle whose exact angle is noisy but whose north/south sign is stable.

We separate the two:
  - MAGNITUDE persistence: corr of advantage SIZE train vs test (the old 0.135)
  - DIRECTION persistence: agreement of argmax-venue train vs test, AND the
    rank-correlation / sign agreement of the advantage ORDERING across venues.

We also test the user's structural map directly: do value-like / momentum-like /
low-risk-like factors have STABLE home venues (small / mid / large)?

Key: we measure direction with estimators that are robust to magnitude noise:
  (a) argmax agreement (hard)
  (b) Spearman rank corr of the V-vector of venue advantages (soft, per factor)
  (c) sign agreement of pairwise venue contrasts (e.g. wide-vs-vw sign)
  (d) shrinkage/pooled direction: average the advantage vector within an
      economic family first (reduces per-factor noise), then test persistence.

If DIRECTION persists (e.g. sign agreement >> 0.5, pooled argmax agree >> 0.25)
while magnitude does not, routing is feasible on the robust coordinate. We then
build a DIRECTION-ONLY router and score it OOS vs coarse=1.302.

Writes cn_direction.json.
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
HERE = Path(__file__).resolve().parent
ANN = np.sqrt(12.0)
COARSE = "wide"

def _sr_series(x):
    x = pd.Series(x).dropna()
    if x.size < 12 or x.std(ddof=1) == 0:
        return np.nan
    return float(ANN * x.mean() / x.std(ddof=1))


def adv_vectors(Bs, disc, venues, months):
    """Advantage vector per factor over `months`: SR in venue v minus SR in
    coarse. Shape (N, V-1) excluding coarse itself."""
    alts = [v for v in venues if v != COARSE]
    A = np.full((len(disc), len(alts)), np.nan)
    base = {f: _sr_series(Bs[COARSE].loc[months, f]) for f in disc}
    for i, f in enumerate(disc):
        for j, v in enumerate(alts):
            A[i, j] = _sr_series(Bs[v].loc[months, f]) - base[f]
    return A, alts


def magnitude_persistence(A_tr, A_te):
    """Old axis: correlation of advantage SIZE across all (i,v) cells."""
    x, y = A_tr.ravel(), A_te.ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def direction_persistence(A_tr, A_te):
    """New axis: how stable is the DIRECTION of the advantage vector?
    (a) argmax agreement; (b) per-factor Spearman rank corr of the venue vector;
    (c) sign agreement of each venue's advantage (is it + or - vs coarse)."""
    from scipy.stats import spearmanr
    N = A_tr.shape[0]
    # (a) hard argmax agreement
    am_tr = np.array([np.nanargmax(A_tr[i]) if np.isfinite(A_tr[i]).any()
                      else -1 for i in range(N)])
    am_te = np.array([np.nanargmax(A_te[i]) if np.isfinite(A_te[i]).any()
                      else -1 for i in range(N)])
    ok = (am_tr >= 0) & (am_te >= 0)
    argmax_agree = float(np.mean(am_tr[ok] == am_te[ok])) if ok.any() else np.nan
    nvenue = A_tr.shape[1]
    argmax_chance = 1.0 / nvenue
    # (b) per-factor rank corr, averaged
    rhos = []
    for i in range(N):
        x, y = A_tr[i], A_te[i]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 3 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
            r, _ = spearmanr(x[m], y[m])
            if np.isfinite(r):
                rhos.append(r)
    rank_corr = float(np.mean(rhos)) if rhos else np.nan
    # (c) sign agreement per venue column
    sign_agree = []
    for j in range(nvenue):
        x, y = A_tr[:, j], A_te[:, j]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 8:
            sign_agree.append(float(np.mean(np.sign(x[m]) == np.sign(y[m]))))
    sign_agree_mean = float(np.mean(sign_agree)) if sign_agree else np.nan
    return {"argmax_agree": argmax_agree, "argmax_chance": argmax_chance,
            "rank_corr_mean": rank_corr, "sign_agree_mean": sign_agree_mean,
            "sign_agree_by_venue": sign_agree}


def pooled_direction(A_tr, A_te, families):
    """Shrinkage: average advantage vectors WITHIN an economic family first, then
    test argmax agreement. Pooling cuts per-factor noise -> isolates whether the
    DIRECTION signal survives once magnitude noise is averaged down."""
    fam_ids = sorted(set(families))
    at, ae = [], []
    for fid in fam_ids:
        idx = [i for i, f in enumerate(families) if f == fid]
        at.append(np.nanmean(A_tr[idx], axis=0))
        ae.append(np.nanmean(A_te[idx], axis=0))
    at, ae = np.array(at), np.array(ae)
    am_tr = np.array([np.nanargmax(r) if np.isfinite(r).any() else -1 for r in at])
    am_te = np.array([np.nanargmax(r) if np.isfinite(r).any() else -1 for r in ae])
    ok = (am_tr >= 0) & (am_te >= 0)
    return {"n_families": len(fam_ids),
            "pooled_argmax_agree": float(np.mean(am_tr[ok] == am_te[ok]))
            if ok.any() else np.nan,
            "chance": 1.0 / A_tr.shape[1]}
def economic_family(name):
    """Economic family from the OSAP factor acronym. Explicit map first (covers
    the hard cases), then keyword fallback. Designed to shrink the 'other' bin."""
    s = name.lower()
    explicit = {
        # payout / issuance / financing
        "netpayoutyield": "payout", "payoutyield": "payout",
        "netdebtprice": "payout", "compequiss": "payout",
        "compositedebtissuance": "payout", "cheq": "payout",
        "delequ": "payout", "delfinl": "payout", "delnetfin": "payout",
        "dellti": "payout", "delcol": "payout", "entmult": "value",
        # value / cash-flow
        "am": "value", "cf": "value", "cashprod": "value", "varcf": "value",
        "netdebtprice2": "value", "equityduration": "value",
        # reversal / momentum
        "lrreversal": "reversal", "mrreversal": "reversal",
        "high52": "momentum", "price": "reversal", "pricedelayrsq": "frictions",
        "pricedelayslope": "frictions", "pricedelaytstat": "frictions",
        # intangibles / R&D / investment
        "rd": "intangibles", "rdability": "intangibles", "adexp": "intangibles",
        "gradexp": "intangibles", "grsaletogroverhead": "intangibles",
        "tang": "intangibles", "hire": "investment", "grcapx": "investment",
        "grcapx3y": "investment", "chnncoa": "investment", "delcoa": "investment",
        # accruals / earnings quality
        "pctacc": "accruals", "chtax": "accruals", "tax": "accruals",
        "earnsupbig": "earnings", "revenuesurprise": "earnings",
        # competition / coskew / liquidity
        "herf": "competition", "herfasset": "competition",
        "coskewness": "risk", "coskewacx": "risk", "frontier": "risk",
        "zerotrade1m": "liquidity", "zerotrade6m": "liquidity",
        "zerotrade12m": "liquidity",
    }
    if s in explicit:
        return explicit[s]
    kw = [
        ("value", ("bm", "ep", "cfp", "sp", "book", "divyield", "ebit", "ey")),
        ("momentum", ("mom", "ret", "52w", "indmom", "resid")),
        ("size", ("size", "mktcap", "me")),
        ("investment", ("inv", "assetgrowth", "ag", "noa", "capex", "grcapx")),
        ("profitability", ("roe", "roa", "gp", "prof", "margin", "fscore", "ato")),
        ("risk", ("beta", "vol", "ivol", "idio", "risk", "std", "skew")),
        ("liquidity", ("turn", "amihud", "illiq", "liq", "volume", "zerotrade")),
        ("payout", ("payout", "issue", "repurchase", "div")),
        ("accruals", ("acc", "accrual")),
    ]
    for fam, keys in kw:
        if any(k in s for k in keys):
            return fam
    return "other"


def direction_router(Bs, disc, venues, A_tr, alts, te, mode="argmax"):
    """Route each factor by the TRAIN direction. mode='argmax' sends factor to
    its train-best alt venue only if that advantage sign is positive, else coarse
    (direction gate). Score EW book OOS vs coarse."""
    venue_of = {}
    for i, f in enumerate(disc):
        row = A_tr[i]
        if np.isfinite(row).any() and np.nanmax(row) > 0:
            venue_of[f] = alts[int(np.nanargmax(row))]
        else:
            venue_of[f] = COARSE
    book = pd.DataFrame({f: Bs[venue_of[f]].loc[te, f]
                         for f in disc}).mean(axis=1).dropna()
    coarse = pd.DataFrame({f: Bs[COARSE].loc[te, f]
                           for f in disc}).mean(axis=1).dropna()
    dist = {}
    for v in set(venue_of.values()):
        dist[v] = int(sum(1 for f in disc if venue_of[f] == v))
    return _sr_series(book), _sr_series(coarse), dist, venue_of


def pooled_router(Bs, disc, venues, A_tr, alts, families, te):
    """Route by the POOLED (family-averaged) train direction: every factor in a
    family goes to that family's pooled-best venue (if positive), else coarse.
    This is the shrinkage router -- the user's 'value->small, mom->mid' map, but
    learned from pooled data instead of per factor."""
    fam_ids = sorted(set(families))
    fam_venue = {}
    for fid in fam_ids:
        idx = [i for i, f in enumerate(disc) if families[i] == fid]
        row = np.nanmean(A_tr[idx], axis=0)
        if np.isfinite(row).any() and np.nanmax(row) > 0:
            fam_venue[fid] = alts[int(np.nanargmax(row))]
        else:
            fam_venue[fid] = COARSE
    venue_of = {f: fam_venue[families[i]] for i, f in enumerate(disc)}
    book = pd.DataFrame({f: Bs[venue_of[f]].loc[te, f]
                         for f in disc}).mean(axis=1).dropna()
    coarse = pd.DataFrame({f: Bs[COARSE].loc[te, f]
                           for f in disc}).mean(axis=1).dropna()
    return _sr_series(book), _sr_series(coarse), fam_venue
def run():
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr, te = idx[idx < mid], idx[idx >= mid]
    disc = [f for f in shared
            if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < 1.96]
    venues = BATTLES
    A_tr, alts = adv_vectors(Bs, disc, venues, tr)
    A_te, _ = adv_vectors(Bs, disc, venues, te)
    families = [economic_family(f) for f in disc]

    mag = magnitude_persistence(A_tr, A_te)
    direc = direction_persistence(A_tr, A_te)
    pooled = pooled_direction(A_tr, A_te, families)

    sr_dir, sr_coarse, dist_dir, venue_of = direction_router(
        Bs, disc, venues, A_tr, alts, te)
    sr_pool, _, fam_venue = pooled_router(
        Bs, disc, venues, A_tr, alts, families, te)

    # family -> pooled best venue map (the user's structural map, data-driven)
    fam_table = {k: v for k, v in fam_venue.items()}

    out = {
        "n_disc": len(disc), "alts": alts, "coarse_SR": sr_coarse,
        "magnitude_persistence": mag,
        "direction_persistence": direc,
        "pooled_direction": pooled,
        "router_direction_SR": sr_dir, "router_direction_dist": dist_dir,
        "router_pooled_SR": sr_pool, "family_venue_map": fam_table,
        "family_counts": {k: int(families.count(k)) for k in set(families)},
        "gate_direction_beats_magnitude": bool(
            np.isfinite(direc["sign_agree_mean"])
            and direc["sign_agree_mean"] > 0.6),
        "gate_router_beats_coarse": bool(
            (np.isfinite(sr_dir) and sr_dir > sr_coarse)
            or (np.isfinite(sr_pool) and sr_pool > sr_coarse)),
    }
    (HERE / "cn_direction.json").write_text(json.dumps(out, indent=2) + "\n")
    _print(out)
    return out


def _print(o):
    print("=" * 76)
    print("DIRECTION vs MAGNITUDE  (does the venue DIRECTION persist?)")
    print("=" * 76)
    print(f"  N_disc={o['n_disc']}  alts={o['alts']}  coarse SR={o['coarse_SR']:.3f}")
    print("-" * 76)
    print(f"  MAGNITUDE persistence (old axis)  phi = "
          f"{o['magnitude_persistence']:+.3f}")
    d = o["direction_persistence"]
    print("  DIRECTION persistence (new axis):")
    print(f"     argmax agree      = {d['argmax_agree']:.3f}  "
          f"(chance {d['argmax_chance']:.3f})")
    print(f"     rank-corr mean    = {d['rank_corr_mean']:+.3f}")
    print(f"     sign agree mean   = {d['sign_agree_mean']:.3f}  "
          f"(chance 0.50)   by venue={[round(x,2) for x in d['sign_agree_by_venue']]}")
    p = o["pooled_direction"]
    print(f"  POOLED direction (family-averaged): argmax agree = "
          f"{p['pooled_argmax_agree']:.3f}  (chance {p['chance']:.3f}, "
          f"n_fam={p['n_families']})")
    print("-" * 76)
    print(f"  DIRECTION-only router SR = {o['router_direction_SR']:+.3f}   "
          f"dist={o['router_direction_dist']}")
    print(f"  POOLED (family) router SR = {o['router_pooled_SR']:+.3f}")
    print(f"  family -> venue map: {o['family_venue_map']}")
    print(f"  family counts: {o['family_counts']}")
    print("-" * 76)
    print(f"  GATE direction robust (sign agree>0.6): "
          f"{o['gate_direction_beats_magnitude']}")
    print(f"  GATE router>coarse: {o['gate_router_beats_coarse']}")
    print("wrote cn_direction.json")


if __name__ == "__main__":
    run()
