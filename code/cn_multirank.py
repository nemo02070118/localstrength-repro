"""MULTI-RANK RESURRECTION  --  break the single-rank (rho~1) wall.

Diagnosis so far: the venue-advantage vectors a_i = u + eps_i are essentially
SINGLE-RANK (rho~1): every factor shares the same small-cap tilt u, so venue
preferences are collinear and cannot be told apart -> all six doors die.

The user's three rescues are three faces of the SAME attack on single-rank:
  (R1) de-mean / venue-orthogonal: subtract the common u (cross-sectional mean
       advantage) from every factor; route on the pure idiosyncratic residual
       eps_i. If a usable 2nd rank exists, routing on eps should beat coarse.
  (R2) venue long-short: trade (best venue - worst venue) advantage directly
       instead of (venue - coarse); u cancels in the difference, exposing eps.
  (R3) multi-rank degree gamma = (sum_{r>=2} lambda_r)/lambda_1 of the venue
       advantage covariance. gamma>0.3 is the pre-registered revival condition.

All three are computed train->test OOS and scored against coarse=1.302 with the
SAME book construction and a paired block bootstrap. Honest gates.

Writes cn_multirank.json.
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
from cn_direction import adv_vectors, economic_family

warnings.filterwarnings("ignore")
HERE = Path(os.environ.get("LOCALSTRENGTH_RESULTS",
                           str(Path(__file__).resolve().parent.parent / "results")))
ANN = np.sqrt(12.0)
COARSE = "wide"

def _sr_series(x):
    x = pd.Series(x).dropna()
    if x.size < 12 or x.std(ddof=1) == 0:
        return np.nan
    return float(ANN * x.mean() / x.std(ddof=1))


def multirank_degree(A):
    """gamma = (sum_{r>=2} lambda_r)/lambda_1 of the (N x V) advantage matrix's
    column covariance (across venues). Also return rho = lambda_1/sum lambda =
    common-tilt share, and the eigenvalue spectrum."""
    good = np.isfinite(A).all(1)
    X = A[good]
    if X.shape[0] < 5:
        return None
    Xc = X - X.mean(0, keepdims=True)
    C = np.cov(Xc, rowvar=False)              # (V x V) across venues
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    ev = ev[ev > 1e-12]
    if ev.size == 0:
        return None
    rho = float(ev[0] / ev.sum())
    gamma = float(ev[1:].sum() / ev[0]) if ev.size > 1 else 0.0
    return {"rho_common_share": rho, "gamma_multirank": gamma,
            "eigs": [float(x) for x in ev], "n_venue": int(A.shape[1])}


def common_idio_split(A_tr, A_te):
    """Split advantage into common u (cross-sectional mean over factors) and
    idiosyncratic eps. Return (u_tr, eps_tr, u_te, eps_te)."""
    u_tr = np.nanmean(A_tr, axis=0, keepdims=True)
    u_te = np.nanmean(A_te, axis=0, keepdims=True)
    return u_tr, A_tr - u_tr, u_te, A_te - u_te


def rho_from_split(A_tr):
    """rho = ||u||^2 / (||u||^2 + mean||eps||^2) -- the common-tilt share used in
    the tau(m) formula."""
    u = np.nanmean(A_tr, axis=0)
    eps = A_tr - u[None, :]
    nu = float(np.sum(u ** 2))
    ne = float(np.nanmean(np.sum(eps ** 2, axis=1)))
    return nu / (nu + ne) if (nu + ne) > 0 else np.nan, nu, ne
def coarse_book(Bs, disc, months):
    return pd.DataFrame({f: Bs[COARSE].loc[months, f]
                         for f in disc}).mean(axis=1).dropna()


def router_from_residual(Bs, disc, alts, eps_tr, te):
    """R1: route each factor to the venue where its IDIOSYNCRATIC (de-meaned)
    advantage is largest and positive; else coarse. Scores the pure 2nd-rank."""
    venue_of = {}
    for i, f in enumerate(disc):
        row = eps_tr[i]
        if np.isfinite(row).any() and np.nanmax(row) > 0:
            venue_of[f] = alts[int(np.nanargmax(row))]
        else:
            venue_of[f] = COARSE
    book = pd.DataFrame({f: Bs[venue_of[f]].loc[te, f]
                         for f in disc}).mean(axis=1).dropna()
    dist = {v: int(sum(1 for f in disc if venue_of[f] == v))
            for v in set(venue_of.values())}
    return book, dist, venue_of


def venue_longshort(Bs, disc, alts, A_tr, te):
    """R2: for each factor, go LONG its train-best venue and SHORT its train-worst
    venue (the within-factor advantage spread). u cancels in the long-short.
    Equal-weight book of these per-factor venue-spread returns."""
    venues = alts + [COARSE]
    streams = []
    for i, f in enumerate(disc):
        # rank venues by train advantage incl coarse (use raw SR per venue)
        srs = {v: _sr_series(Bs[v].loc[te.min():te.max(), f]) for v in venues}
        row = A_tr[i]
        if not np.isfinite(row).any():
            continue
        vlong = alts[int(np.nanargmax(row))]
        vshort = alts[int(np.nanargmin(row))]
        if vlong == vshort:
            continue
        s = (Bs[vlong].loc[te, f] - Bs[vshort].loc[te, f])
        streams.append(s)
    if not streams:
        return pd.Series(dtype=float)
    return pd.concat(streams, axis=1).mean(axis=1).dropna()
def boot_diff(router, coarse, n_boot=3000, block=6, seed=0):
    both = pd.concat([router.rename("r"), coarse.rename("c")], axis=1).dropna()
    if both.shape[0] < 24:
        return None
    r, c = both["r"].to_numpy(), both["c"].to_numpy()
    n = len(r)
    point = _sr_series(pd.Series(r)) - _sr_series(pd.Series(c))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        diffs[b] = _sr_series(pd.Series(r[idx])) - _sr_series(pd.Series(c[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"point": float(point), "ci95": [float(lo), float(hi)],
            "p_gt0": float(np.mean(diffs > 0)), "n": int(n)}


def boot_absolute(series, n_boot=3000, block=6, seed=1):
    """Bootstrap p that an ABSOLUTE SR (e.g. venue long-short) > 0."""
    x = pd.Series(series).dropna().to_numpy()
    if len(x) < 24:
        return None
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    srs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        srs[b] = _sr_series(pd.Series(x[idx]))
    lo, hi = np.percentile(srs, [2.5, 97.5])
    return {"sr": _sr_series(pd.Series(x)), "ci95": [float(lo), float(hi)],
            "p_gt0": float(np.mean(srs > 0)), "n": int(n)}
def run():
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr, te = idx[idx < mid], idx[idx >= mid]
    disc = [f for f in shared
            if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < 1.96]
    A_tr, alts = adv_vectors(Bs, disc, BATTLES, tr)
    A_te, _ = adv_vectors(Bs, disc, BATTLES, te)

    # R3: multi-rank degree + rho
    mr_tr = multirank_degree(A_tr)
    mr_te = multirank_degree(A_te)
    rho, nu, ne = rho_from_split(A_tr)
    u_tr, eps_tr, u_te, eps_te = common_idio_split(A_tr, A_te)
    # persistence of the idiosyncratic (2nd-rank) direction
    x, y = eps_tr.ravel(), eps_te.ravel()
    m = np.isfinite(x) & np.isfinite(y)
    eps_persist = (float(np.corrcoef(x[m], y[m])[0, 1])
                   if m.sum() > 8 and np.std(x[m]) > 0 else np.nan)

    coarse = coarse_book(Bs, disc, te)
    sr_coarse = _sr_series(coarse)

    # R1: residual (de-meaned) router
    book_r1, dist_r1, _ = router_from_residual(Bs, disc, alts, eps_tr, te)
    sr_r1 = _sr_series(book_r1)
    boot_r1 = boot_diff(book_r1, coarse)

    # R2: venue long-short (absolute SR, this is a NEW alpha stream, not vs coarse)
    ls = venue_longshort(Bs, disc, alts, A_tr, te)
    sr_ls = _sr_series(ls)
    boot_ls = boot_absolute(ls)

    out = {
        "n_disc": len(disc), "alts": alts, "coarse_SR": sr_coarse,
        "multirank_train": mr_tr, "multirank_test": mr_te,
        "rho_split": {"rho": rho, "norm_u2": nu, "mean_norm_eps2": ne},
        "eps_direction_persistence": eps_persist,
        "R1_residual_router": {"SR": sr_r1, "dist": dist_r1, "boot": boot_r1},
        "R2_venue_longshort": {"SR": sr_ls, "boot": boot_ls},
        "gate_gamma_gt_030": bool(mr_tr and mr_tr["gamma_multirank"] > 0.30),
        "gate_R1_beats_coarse": bool(
            boot_r1 is not None and boot_r1["p_gt0"] >= 0.95),
        "gate_R2_positive_alpha": bool(
            boot_ls is not None and boot_ls["p_gt0"] >= 0.95),
    }
    (HERE / "cn_multirank.json").write_text(json.dumps(out, indent=2) + "\n")
    _print(out)
    return out


def _print(o):
    print("=" * 76)
    print("MULTI-RANK RESURRECTION  (break the rho~1 wall)")
    print("=" * 76)
    print(f"  N_disc={o['n_disc']}  alts={o['alts']}  coarse SR={o['coarse_SR']:.3f}")
    mt = o["multirank_train"]
    print("-" * 76)
    print("  R3  MULTI-RANK DEGREE (venue advantage covariance):")
    print(f"     eigenvalues (train) = {[round(x,4) for x in mt['eigs']]}")
    print(f"     rho (lambda1 share)  = {mt['rho_common_share']:.3f}")
    print(f"     gamma (rank>=2 / l1) = {mt['gamma_multirank']:.3f}  "
          f"(revival needs >0.30)")
    rs = o["rho_split"]
    print(f"     rho via u/eps split  = {rs['rho']:.3f}  "
          f"(||u||^2={rs['norm_u2']:.4f}, mean||eps||^2={rs['mean_norm_eps2']:.4f})")
    print(f"     eps-direction persistence = {o['eps_direction_persistence']:+.3f}")
    print("-" * 76)
    r1 = o["R1_residual_router"]
    print(f"  R1  RESIDUAL (de-meaned) router SR = {r1['SR']:+.4f}  dist={r1['dist']}")
    if r1["boot"]:
        b = r1["boot"]
        print(f"      vs coarse: diff={b['point']:+.4f} CI[{b['ci95'][0]:+.3f},"
              f"{b['ci95'][1]:+.3f}] p(>0)={b['p_gt0']:.3f}")
    r2 = o["R2_venue_longshort"]
    print(f"  R2  VENUE LONG-SHORT alpha SR = {r2['SR']:+.4f}")
    if r2["boot"]:
        b = r2["boot"]
        print(f"      absolute: CI[{b['ci95'][0]:+.3f},{b['ci95'][1]:+.3f}] "
              f"p(SR>0)={b['p_gt0']:.3f}  (n={b['n']})")
    print("-" * 76)
    print(f"  GATE gamma>0.30 (multi-rank):    {o['gate_gamma_gt_030']}")
    print(f"  GATE R1 residual > coarse:       {o['gate_R1_beats_coarse']}")
    print(f"  GATE R2 venue-LS alpha > 0:      {o['gate_R2_positive_alpha']}")
    print("wrote cn_multirank.json")


if __name__ == "__main__":
    run()
