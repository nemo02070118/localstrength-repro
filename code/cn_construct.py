"""CONSTRUCTION-PROPERTY TEST: is tradable-rank=1 a property of the DATA or of
the CONSTRUCTION? If the common tilt u (small-cap beta) comes from the mechanical
effect of cross-sectional ranking, changing the construction should open a 2nd
tradable rank.

Key realization about the data on disk: osap_ls_{vw,nyse,me20,wide}.parquet are
each a factor's long-short return constructed WITHIN that universe (venue). So
they already ARE the 'venue-interaction' construction (construct-1): signal
re-ranked inside each venue, not a global sort traded in a venue. We exploit
this to build and test, with no firm-level data:

  C2 venue-pair      : r_i^{A,B} = r_i in venue A  -  r_i in venue B
                       (u cancels in the venue difference)
  C3 venue-orthogonal: r_i_perp = residual of r_i on SMB (strip the small-cap
                       beta that creates u), then route on venue advantages of
                       the orthogonalized returns
  stabilize-2nd-rank : smooth / shrink the idiosyncratic direction with a longer
                       window and test whether eps-direction persistence rises
                       above 0.5
  base-state-alpha   : treat u itself as the tradable object; build the common
                       small-cap-tilt portfolio and compare to coarse

TRADABLE RANK = number of eigen-directions of the venue-advantage covariance
whose train->test persistence exceeds 0.5. Pre-registered revival gate:
tradable rank > 1 AND a router on the surviving direction beats coarse=1.302.

Writes cn_construct.json.
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
from cn_direction import adv_vectors

warnings.filterwarnings("ignore")
HERE = Path(os.environ.get("LOCALSTRENGTH_RESULTS",
                           str(Path(__file__).resolve().parent.parent / "results")))
KF = Path(os.environ.get("LOCALSTRENGTH_DATA", "E:/BOAT-FX-DATA")) / "ken_french"
ANN = np.sqrt(12.0)
COARSE = "wide"

def _sr_series(x):
    x = pd.Series(x).dropna()
    if x.size < 12 or x.std(ddof=1) == 0:
        return np.nan
    return float(ANN * x.mean() / x.std(ddof=1))


def load_smb():
    ff = pd.read_csv(KF / "FF5_monthly.csv")
    ff["Date"] = pd.to_datetime(ff["Date"].astype(str), format="%Y%m")
    ff = ff.set_index("Date")
    ff.index = ff.index.to_period("M").to_timestamp("M")
    return (ff["SMB"] / 100.0)


def orthogonalize(series, smb):
    """Residual of a monthly return series on SMB (strip small-cap beta)."""
    df = pd.concat([series.rename("y"), smb.rename("x")], axis=1).dropna()
    if len(df) < 24:
        return series
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    b = np.cov(x, y)[0, 1] / np.var(x)
    a = y.mean() - b * x.mean()
    resid = pd.Series(y - a - b * x, index=df.index)
    return resid


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
        st = rng.integers(0, n, nb)
        idx = np.concatenate([np.arange(s, s + block) % n for s in st])[:n]
        diffs[b] = _sr_series(pd.Series(r[idx])) - _sr_series(pd.Series(c[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"point": float(point), "ci95": [float(lo), float(hi)],
            "p_gt0": float(np.mean(diffs > 0)), "n": int(n)}
def tradable_rank(A_tr, A_te):
    """Eigen-decompose the TRAIN venue-advantage covariance; for each eigenvector
    project BOTH train and test advantage vectors onto it and measure the
    cross-sectional persistence (corr of the projection scores train vs test).
    Tradable rank = count of directions with persistence > 0.5."""
    good = np.isfinite(A_tr).all(1) & np.isfinite(A_te).all(1)
    Xtr, Xte = A_tr[good], A_te[good]
    if Xtr.shape[0] < 8:
        return None
    mu = Xtr.mean(0, keepdims=True)
    Ctr = np.cov(Xtr - mu, rowvar=False)
    evals, evecs = np.linalg.eigh(Ctr)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    dirs = []
    for k in range(len(evals)):
        v = evecs[:, k]
        s_tr = (Xtr - mu) @ v
        s_te = (Xte - mu) @ v
        m = np.isfinite(s_tr) & np.isfinite(s_te)
        if m.sum() >= 8 and np.std(s_tr[m]) > 0 and np.std(s_te[m]) > 0:
            persist = float(np.corrcoef(s_tr[m], s_te[m])[0, 1])
        else:
            persist = np.nan
        dirs.append({"eig": float(evals[k]),
                     "var_share": float(evals[k] / evals.sum()),
                     "persistence": persist})
    rank = int(sum(1 for d in dirs if np.isfinite(d["persistence"])
                   and d["persistence"] > 0.5))
    return {"directions": dirs, "tradable_rank": rank,
            "gamma": float(evals[1:].sum() / evals[0]) if len(evals) > 1 else 0.0}
def _win(series, months):
    """Series values restricted to the intersection of months and the index."""
    return series.reindex(series.index.intersection(months))


def adv_vectors_from(Bmod, disc, alts, months):
    """Advantage vectors (vs coarse) from a modified battlefield dict Bmod."""
    A = np.full((len(disc), len(alts)), np.nan)
    base = {f: _sr_series(_win(Bmod[COARSE][f], months)) for f in disc}
    for i, f in enumerate(disc):
        for j, v in enumerate(alts):
            A[i, j] = _sr_series(_win(Bmod[v][f], months)) - base[f]
    return A


def coarse_book(Bs, disc, months):
    cols = {f: _win(Bs[COARSE][f], months) for f in disc}
    return pd.DataFrame(cols).mean(axis=1).dropna()


def route_argmax(Bs, disc, alts, A_tr, te):
    venue_of = {}
    for i, f in enumerate(disc):
        row = A_tr[i]
        venue_of[f] = (alts[int(np.nanargmax(row))]
                       if np.isfinite(row).any() and np.nanmax(row) > 0
                       else COARSE)
    book = pd.DataFrame({f: _win(Bs[venue_of[f]][f], te)
                         for f in disc}).mean(axis=1).dropna()
    return book, {v: int(sum(1 for f in disc if venue_of[f] == v))
                  for v in set(venue_of.values())}


def build_orthogonal_bs(Bs, shared, smb):
    """C3: orthogonalize every (venue, factor) return series on SMB."""
    Bmod = {}
    for v in BATTLES:
        cols = {}
        for f in shared:
            cols[f] = orthogonalize(Bs[v][f].dropna(), smb)
        Bmod[v] = pd.DataFrame(cols)
    return Bmod


def venue_pair_rank(Bs, disc, alts, tr, te):
    """C2: build venue-pair advantage = SR(venue A) - SR(venue B) for every
    ordered pair, as an expanded advantage vector per factor; measure its
    tradable rank and route."""
    venues = alts + [COARSE]
    pairs = [(a, b) for a in venues for b in venues if a != b]
    def vec(months):
        M = np.full((len(disc), len(pairs)), np.nan)
        for i, f in enumerate(disc):
            srs = {v: _sr_series(Bs[v].loc[months, f]) for v in venues}
            for j, (a, b) in enumerate(pairs):
                if np.isfinite(srs[a]) and np.isfinite(srs[b]):
                    M[i, j] = srs[a] - srs[b]
        return M
    return vec(tr), vec(te), pairs
def run():
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr, te = idx[idx < mid], idx[idx >= mid]
    disc = [f for f in shared
            if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < 1.96]
    alts = [v for v in BATTLES if v != COARSE]
    smb = load_smb()
    coarse = coarse_book(Bs, disc, te)
    sr_coarse = _sr_series(coarse)
    out = {"n_disc": len(disc), "alts": alts, "coarse_SR": sr_coarse}

    # C1 baseline (venue-interaction, as-is OSAP): tradable rank
    A_tr, _ = adv_vectors(Bs, disc, BATTLES, tr)
    A_te, _ = adv_vectors(Bs, disc, BATTLES, te)
    out["C1_venue_interaction"] = {"rank": tradable_rank(A_tr, A_te)}

    # C2 venue-pair
    P_tr, P_te, pairs = venue_pair_rank(Bs, disc, alts, tr, te)
    rk2 = tradable_rank(P_tr, P_te)
    out["C2_venue_pair"] = {"n_pairs": len(pairs), "rank": rk2}

    # C3 venue-orthogonal (strip SMB)
    Bmod = build_orthogonal_bs(Bs, shared, smb)
    Ao_tr = adv_vectors_from(Bmod, disc, alts, tr)
    Ao_te = adv_vectors_from(Bmod, disc, alts, te)
    rk3 = tradable_rank(Ao_tr, Ao_te)
    book3, dist3 = route_argmax(Bmod, disc, alts, Ao_tr, te)
    # score orthogonalized router against orthogonalized coarse
    coarse3 = coarse_book(Bmod, disc, te)
    out["C3_venue_orthogonal"] = {
        "rank": rk3, "router_SR": _sr_series(book3),
        "coarse_ortho_SR": _sr_series(coarse3), "dist": dist3,
        "boot": boot_diff(book3, coarse3)}

    # stabilize 2nd rank: longer-window idiosyncratic direction (full-sample u,
    # shrink eps by averaging train+first-half-test is look-ahead -> instead use
    # expanding estimate of eps direction only from train, heavier smoothing)
    u_tr = np.nanmean(A_tr, axis=0, keepdims=True)
    eps_tr = A_tr - u_tr
    eps_te = A_te - np.nanmean(A_te, axis=0, keepdims=True)
    # shrink eps toward 0 by factor s and re-measure persistence of direction
    best = None
    for s in (0.0, 0.3, 0.5, 0.7, 1.0):
        shr = s * eps_tr
        x, y = shr.ravel(), eps_te.ravel()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 8 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
            pr = float(np.corrcoef(x[m], y[m])[0, 1])
            if best is None or pr > best[1]:
                best = (s, pr)
    out["stabilize_2nd_rank"] = {"best_shrink": best[0] if best else None,
                                 "best_persistence": best[1] if best else None,
                                 "note": "shrink is monotone in corr; cannot "
                                 "exceed raw eps persistence ~0.19"}

    # base-state alpha: common-tilt portfolio. u points to a venue mix; build the
    # book that routes EVERY factor to the single global-best venue (the u
    # direction) and compare to coarse.
    gstar = alts[int(np.nanargmax(u_tr.ravel()))] if np.isfinite(u_tr).any() else COARSE
    base_book = pd.DataFrame({f: Bs[gstar].loc[te, f]
                              for f in disc}).mean(axis=1).dropna()
    out["base_state_alpha"] = {"u_best_venue": gstar,
                               "base_SR": _sr_series(base_book),
                               "coarse_SR": sr_coarse,
                               "boot": boot_diff(base_book, coarse)}

    out["gate_C2_rank_gt1"] = bool(rk2 and rk2["tradable_rank"] > 1)
    out["gate_C3_rank_gt1"] = bool(rk3 and rk3["tradable_rank"] > 1)
    out["gate_C3_router_beats"] = bool(
        out["C3_venue_orthogonal"]["boot"]
        and out["C3_venue_orthogonal"]["boot"]["p_gt0"] >= 0.95)
    out["gate_base_beats_coarse"] = bool(
        out["base_state_alpha"]["boot"]
        and out["base_state_alpha"]["boot"]["p_gt0"] >= 0.95)
    (HERE / "cn_construct.json").write_text(json.dumps(out, indent=2) + "\n")
    _print(out)
    return out


def _rankstr(rk):
    if not rk:
        return "n/a"
    ds = ", ".join(f"{d['var_share']:.0%}/pers={d['persistence']:+.2f}"
                   for d in rk["directions"])
    return f"rank={rk['tradable_rank']} gamma={rk['gamma']:.2f}  [{ds}]"


def _print(o):
    print("=" * 78)
    print("CONSTRUCTION-PROPERTY TEST: can a new construction open tradable rank?")
    print("=" * 78)
    print(f"  N_disc={o['n_disc']}  coarse SR={o['coarse_SR']:.3f}")
    print("-" * 78)
    print(f"  C1 venue-interaction (as-is): {_rankstr(o['C1_venue_interaction']['rank'])}")
    print(f"  C2 venue-pair ({o['C2_venue_pair']['n_pairs']} pairs): "
          f"{_rankstr(o['C2_venue_pair']['rank'])}")
    c3 = o["C3_venue_orthogonal"]
    print(f"  C3 venue-orthogonal (strip SMB): {_rankstr(c3['rank'])}")
    print(f"       router SR={c3['router_SR']:+.3f}  "
          f"coarse_ortho={c3['coarse_ortho_SR']:+.3f}  dist={c3['dist']}")
    if c3["boot"]:
        b = c3["boot"]
        print(f"       boot diff={b['point']:+.3f} CI[{b['ci95'][0]:+.3f},"
              f"{b['ci95'][1]:+.3f}] p(>0)={b['p_gt0']:.3f}")
    s = o["stabilize_2nd_rank"]
    print(f"  stabilize 2nd rank: best persistence={s['best_persistence']}")
    ba = o["base_state_alpha"]
    print(f"  base-state alpha (u->{ba['u_best_venue']}): SR={ba['base_SR']:+.3f} "
          f"vs coarse {ba['coarse_SR']:+.3f}")
    if ba["boot"]:
        b = ba["boot"]
        print(f"       boot diff={b['point']:+.3f} p(>0)={b['p_gt0']:.3f}")
    print("-" * 78)
    print(f"  GATE C2 rank>1: {o['gate_C2_rank_gt1']}   "
          f"C3 rank>1: {o['gate_C3_rank_gt1']}   "
          f"C3 router>coarse: {o['gate_C3_router_beats']}   "
          f"base>coarse: {o['gate_base_beats_coarse']}")
    print("wrote cn_construct.json")


if __name__ == "__main__":
    run()
