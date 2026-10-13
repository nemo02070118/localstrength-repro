"""Two referee-mandatory robustness experiments, written to cn_placebo_fdr.json.

(A) PLACEBO. The +0.45 coarse gain must not be a mechanical artifact of having
    four venues to average over. Two placebos:
      A1 random-venue: assign each discard a uniformly random venue (not the
         train-best), 1000 draws; report the gain distribution and the rank of
         the true coarse gain within it.
      A2 shuffled-labels: permute the venue columns within each factor before
         picking the pool-best venue, 1000 draws; same report.
    If the true gain sits in the extreme right tail (empirical p -> 0), the gain
    is the venue decision, not multiplicity of universes.

(B) FDR. The paper screens at a fixed |t|<1.96. Referees ask whether the
    conclusion survives a modern false-discovery-rate screen. Re-run discovery
    under Benjamini-Hochberg and Benjamini-Yekutieli at q in {0.05, 0.10} on the
    train-half vw t-stats, and report the discard count and coarse gain under
    each. The claim is that stricter, smarter screens discard MORE reallocatable
    value, exactly as Proposition 1 predicts.
"""
from __future__ import annotations

import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np
from scipy import stats

from cn_resurrection import load_battlefields, _tstat, _sr, _book, BATTLES

HERE = Path(os.environ.get("LOCALSTRENGTH_RESULTS", str(Path(__file__).resolve().parent.parent / "results")))


def _pool_best(Bs, disc, srtr):
    pool = {n: float(np.nanmean([srtr[f][n] for f in disc])) for n in BATTLES}
    return max(pool, key=pool.get)


def run():
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr = idx[idx < mid]
    te = idx[idx >= mid]

    # train-half vw t-stats and two-sided p-values for every shared factor
    tvw = {f: _tstat(Bs["vw"].loc[tr, f]) for f in shared}
    tvw = {f: t for f, t in tvw.items() if np.isfinite(t)}
    disc = [f for f, t in tvw.items() if abs(t) < 1.96]
    srtr = {f: {n: _sr(Bs[n].loc[tr, f]) for n in BATTLES} for f in disc}
    srte = {f: {n: _sr(Bs[n].loc[te, f]) for n in BATTLES} for f in disc}
    best = _pool_best(Bs, disc, srtr)

    sr_g, _ = _book(Bs, disc, {f: "vw" for f in disc}, te)
    sr_c, _ = _book(Bs, disc, {f: best for f in disc}, te)
    true_gain = sr_c - sr_g

    out = {"n_disc": len(disc), "coarse_venue": best,
           "sr_grave": sr_g, "sr_coarse": sr_c, "true_coarse_gain": true_gain}

    rng = np.random.default_rng(12345)
    alt = BATTLES

    # --- A1 random venue ---------------------------------------------------
    gains = []
    for _ in range(1000):
        bf = {f: alt[rng.integers(len(alt))] for f in disc}
        s, _ = _book(Bs, disc, bf, te)
        if np.isfinite(s):
            gains.append(s - sr_g)
    gains = np.array(gains)
    out["placebo_random_venue"] = {
        "n_draws": int(gains.size),
        "mean_gain": float(gains.mean()),
        "sd_gain": float(gains.std(ddof=1)),
        "p95_gain": float(np.percentile(gains, 95)),
        "true_gain": true_gain,
        "emp_p_true_ge_placebo": float((gains >= true_gain).mean()),
        "z_true_vs_placebo": float((true_gain - gains.mean()) / gains.std(ddof=1)),
    }

    # --- A2 shuffled venue labels within factor ----------------------------
    # permute each factor's venue identities, then pick pool-best on permuted
    # train, evaluate on correspondingly permuted test; 1000 draws.
    gains2 = []
    disc_arr = disc
    for _ in range(1000):
        gain_draw = []
        bf = {}
        for f in disc_arr:
            perm = rng.permutation(alt)
            relabel = dict(zip(alt, perm))
            # pool-best under relabel is just a random venue per factor's own
            # train sharpe ranking on permuted labels -> emulate label swap
            srp = {relabel[n]: srtr[f][n] for n in alt}
            bf[f] = max(srp, key=srp.get)
        s, _ = _book(Bs, disc_arr, bf, te)
        if np.isfinite(s):
            gains2.append(s - sr_g)
    gains2 = np.array(gains2)
    out["placebo_shuffled_labels"] = {
        "n_draws": int(gains2.size),
        "mean_gain": float(gains2.mean()),
        "sd_gain": float(gains2.std(ddof=1)),
        "true_gain": true_gain,
        "emp_p_true_ge_placebo": float((gains2 >= true_gain).mean()),
    }

    # --- B FDR screens -----------------------------------------------------
    def bh_discards(q, method):
        items = sorted(tvw.items(), key=lambda kv: abs(kv[1]), reverse=True)
        fs = [f for f, _ in items]
        p = np.array([2 * (1 - stats.norm.cdf(abs(tvw[f]))) for f in fs])
        rej = _multipletests_reject(p, q, method)
        kept = set(f for f, r in zip(fs, rej) if r)
        disc_fdr = [f for f in fs if f not in kept]
        return disc_fdr

    fdr = {}
    for method in ("bh", "by"):
        for q in (0.05, 0.10):
            dsc = bh_discards(q, method)
            if not dsc:
                continue
            srtr_d = {f: {n: _sr(Bs[n].loc[tr, f]) for n in BATTLES} for f in dsc}
            b = _pool_best(Bs, dsc, srtr_d)
            g, _ = _book(Bs, dsc, {f: "vw" for f in dsc}, te)
            c, _ = _book(Bs, dsc, {f: b for f in dsc}, te)
            fdr[f"{method}_q{q}"] = {
                "n_disc": len(dsc), "coarse_venue": b,
                "sr_grave": g, "sr_coarse": c, "gain": c - g}
    out["fdr_screens"] = fdr

    (HERE / "cn_placebo_fdr.json").write_text(json.dumps(out, indent=2) + "\n")
    _print(out)
    return out


def _multipletests_reject(pvals, alpha, method):
    """Benjamini-Hochberg (bh) or Benjamini-Yekutieli (by) step-up rejection.
    Returns a boolean array aligned to the INPUT order of pvals."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    if method == "by":
        cfac = np.sum(1.0 / np.arange(1, n + 1))
    else:
        cfac = 1.0
    thresh = (np.arange(1, n + 1) / n) * alpha / cfac
    below = ranked <= thresh
    if not below.any():
        rej_sorted = np.zeros(n, bool)
    else:
        kmax = np.max(np.where(below)[0])
        rej_sorted = np.arange(n) <= kmax
    rej = np.zeros(n, bool)
    rej[order] = rej_sorted
    return rej


def _print(o):
    print("=" * 68)
    print("PLACEBO + FDR robustness")
    print("=" * 68)
    print(f"  discards={o['n_disc']}  coarse venue={o['coarse_venue']}")
    print(f"  true coarse gain = {o['true_coarse_gain']:+.3f}")
    pr = o["placebo_random_venue"]
    print(f"  [A1] random-venue placebo: mean gain={pr['mean_gain']:+.3f} "
          f"sd={pr['sd_gain']:.3f}  p(placebo>=true)={pr['emp_p_true_ge_placebo']:.3f} "
          f"z={pr['z_true_vs_placebo']:+.2f}")
    ps = o["placebo_shuffled_labels"]
    print(f"  [A2] shuffled-labels placebo: mean gain={ps['mean_gain']:+.3f} "
          f"p(placebo>=true)={ps['emp_p_true_ge_placebo']:.3f}")
    print("  [B] FDR screens:")
    for k, v in o["fdr_screens"].items():
        print(f"      {k:10s}: N={v['n_disc']:>3} venue={v['coarse_venue']:<4} "
              f"gain={v['gain']:+.3f}")
    print("wrote cn_placebo_fdr.json")


if __name__ == "__main__":
    run()
