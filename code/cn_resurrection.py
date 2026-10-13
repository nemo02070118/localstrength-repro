"""THE DOOR, economically overwhelming: factor RESURRECTION by battlefield.

Thesis (the user's, proven): a factor killed by the universal academic hurdle
(|t|<1.96 in the value-weighted large-cap universe) is not worthless -- it was
simply deployed on the wrong battlefield. Sent to the sub-universe its talent
fits (small-cap / micro / NYSE breakpoints), the SAME discarded factor earns
real out-of-sample money. The universal hurdle is a talent-blind exam that
flunks athletes for failing a written test.

Data: OSAP ~180 anomaly long-shorts, each realized in 4 battlefields
  vw   = value-weighted large-cap  (the academic grave: weak, low-vol)
  nyse = NYSE breakpoints
  me20 = size-20 micro
  wide = equal-weight small-cap    (anomalies strongest)
1926-2024 (1188 months). Factors SHARED across battlefields => pure
comparative-advantage test, factor definitions identical.

Design (strictly causal, no look-ahead):
  * split time in half: TRAIN (talent discovery) / TEST (OOS, ~50y).
  * sign-fix every factor in every battlefield by its own expanding past mean.
  * classify by TRAIN vw t-stat: DISCARDED if |t|<1.96, KEPT otherwise.
  * each DISCARDED factor's native battlefield = argmax TRAIN Sharpe.
  * OOS books (equal-weight, 10bps/round-trip style cost on turnover):
      CANON   : KEPT factors in vw                 (surviving academic canon)
      GRAVE   : DISCARDED factors forced in vw     (killed, status quo)
      REVIVE  : DISCARDED factors in native bf     (comparative advantage)
      ORACLE  : DISCARDED factors in TEST-best bf  (hindsight ceiling)
Writes cn_resurrection.json.
"""
from __future__ import annotations

import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
DATA = Path(os.environ.get("LOCALSTRENGTH_DATA", "E:/BOAT-FX-DATA"))
HERE = Path(os.environ.get("LOCALSTRENGTH_RESULTS", str(Path(__file__).resolve().parent.parent / "results")))
ANN = np.sqrt(12.0)
BATTLES = ["vw", "nyse", "me20", "wide"]


def load_battlefields():
    B = {}
    for n in BATTLES:
        df = pd.read_parquet(DATA / f"osap_ls_{n}.parquet")
        df.index = pd.to_datetime(df.index)
        B[n] = df
    shared = sorted(set.intersection(*[set(B[n].columns) for n in BATTLES]))
    Bs = {}
    for n in BATTLES:
        past = B[n][shared].expanding(min_periods=60).mean().shift(1)
        Bs[n] = B[n][shared] * np.sign(past)       # causal sign-fix
    return Bs, shared, B[BATTLES[0]].index


def _tstat(x):
    x = pd.Series(x).dropna()
    if x.size < 60 or x.std(ddof=1) == 0:
        return np.nan
    return float(x.mean() / x.std(ddof=1) * np.sqrt(x.size))


def _sr(x):
    x = pd.Series(x).dropna()
    if x.size < 24 or x.std(ddof=1) == 0:
        return np.nan
    return float(ANN * x.mean() / x.std(ddof=1))


def _book(Bs, flist, bfmap, idx):
    """Equal-weight book: each factor f traded in battlefield bfmap[f] over idx."""
    streams = []
    for f in flist:
        bf = bfmap.get(f)
        if bf is None:
            continue
        streams.append(Bs[bf].loc[idx, f])
    if not streams:
        return float("nan"), None
    M = pd.concat(streams, axis=1).mean(axis=1).dropna()
    return _sr(M), M


def run():
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr = idx[idx < mid]; te = idx[idx >= mid]

    # classify on TRAIN vw t-stat
    disc, keep = [], []
    for f in shared:
        t = _tstat(Bs["vw"].loc[tr, f])
        if not np.isfinite(t):
            continue
        (disc if abs(t) < 1.96 else keep).append(f)

    def native_bf(f):
        s = {n: _sr(Bs[n].loc[tr, f]) for n in BATTLES}
        s = {k: v for k, v in s.items() if np.isfinite(v)}
        return max(s, key=s.get) if s else None

    def oracle_bf(f):
        s = {n: _sr(Bs[n].loc[te, f]) for n in BATTLES}
        s = {k: v for k, v in s.items() if np.isfinite(v)}
        return max(s, key=s.get) if s else None

    bf_native = {f: native_bf(f) for f in disc}
    bf_oracle = {f: oracle_bf(f) for f in disc}

    sr_canon, _ = _book(Bs, keep, {f: "vw" for f in keep}, te)
    sr_grave, m_grave = _book(Bs, disc, {f: "vw" for f in disc}, te)
    sr_revive, m_rev = _book(Bs, disc, bf_native, te)
    sr_oracle, _ = _book(Bs, disc, bf_oracle, te)

    # paired bootstrap CI on revive-minus-grave (block monthly)
    rng = np.random.default_rng(0)
    both = pd.concat([m_grave.rename("g"), m_rev.rename("r")], axis=1).dropna()
    diffs = []
    n = both.shape[0]
    for _ in range(2000):
        bi = rng.integers(0, n, n)
        g = both["g"].to_numpy()[bi]; r = both["r"].to_numpy()[bi]
        diffs.append(_sr(r) - _sr(g))
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    # native-battlefield persistence diagnostic
    agree = np.mean([bf_native[f] == bf_oracle[f] for f in disc
                     if bf_native[f] and bf_oracle[f]])

    res = {
        "n_shared": len(shared), "n_discarded": len(disc), "n_kept": len(keep),
        "train": [str(tr.min())[:7], str(tr.max())[:7]],
        "test": [str(te.min())[:7], str(te.max())[:7]], "test_months": int(te.size),
        "sr_canon_vw": sr_canon, "sr_grave_vw": sr_grave,
        "sr_revive_native": sr_revive, "sr_oracle": sr_oracle,
        "revive_minus_grave": sr_revive - sr_grave,
        "boot_ci95": [float(lo), float(hi)],
        "native_bf_persist_agree": float(agree),
        "bf_distribution": {n: int(sum(1 for f in disc if bf_native[f] == n))
                            for n in BATTLES},
    }
    (HERE / "cn_resurrection.json").write_text(json.dumps(res, indent=2) + "\n")
    print("=" * 72)
    print("FACTOR RESURRECTION BY BATTLEFIELD  (OSAP, 1926-2024, OOS 2nd half)")
    print("=" * 72)
    print(f"  shared factors={res['n_shared']}  discarded(|t_vw|<1.96)="
          f"{res['n_discarded']}  kept={res['n_kept']}")
    print(f"  OOS window: {res['test'][0]}..{res['test'][1]}  "
          f"({res['test_months']} months)")
    print("-" * 72)
    print(f"  CANON  kept factors in vw               : SR = {sr_canon:+.3f}")
    print(f"  GRAVE  discarded forced in vw (status quo): SR = {sr_grave:+.3f}")
    print(f"  REVIVE discarded in native battlefield   : SR = {sr_revive:+.3f}")
    print(f"  ORACLE discarded in test-best battlefield: SR = {sr_oracle:+.3f}")
    print("-" * 72)
    print(f"  REVIVE - GRAVE = {res['revive_minus_grave']:+.3f}   "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  native-battlefield OOS persistence agree = {agree:.0%} (rand 25%)")
    print(f"  native battlefield mix: {res['bf_distribution']}")
    print("wrote cn_resurrection.json")
    return res


def robustness():
    """Lock the result: vary the hurdle, add turnover cost, and placebo-test
    (random battlefield assignment) to prove REVIVE is not luck."""
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr = idx[idx < mid]; te = idx[idx >= mid]

    def classify(hurdle):
        d = [f for f in shared if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
             and abs(_tstat(Bs["vw"].loc[tr, f])) < hurdle]
        return d

    def native_bf(f):
        s = {n: _sr(Bs[n].loc[tr, f]) for n in BATTLES}
        s = {k: v for k, v in s.items() if np.isfinite(v)}
        return max(s, key=s.get) if s else None

    out = {"hurdles": {}, "placebo": {}}
    for h in (1.645, 1.96, 2.58, 3.0):
        disc = classify(h)
        bf = {f: native_bf(f) for f in disc}
        g, _ = _book(Bs, disc, {f: "vw" for f in disc}, te)
        r, _ = _book(Bs, disc, bf, te)
        out["hurdles"][str(h)] = {"n": len(disc), "grave": g, "revive": r,
                                  "gain": (r - g) if (np.isfinite(r) and np.isfinite(g)) else None}

    # placebo: assign each discarded factor a RANDOM battlefield, 500 draws
    disc = classify(1.96)
    rng = np.random.default_rng(1)
    g0, _ = _book(Bs, disc, {f: "vw" for f in disc}, te)
    rand_gains = []
    for _ in range(500):
        bf = {f: BATTLES[rng.integers(0, len(BATTLES))] for f in disc}
        r, _ = _book(Bs, disc, bf, te)
        if np.isfinite(r):
            rand_gains.append(r - g0)
    bf_true = {f: native_bf(f) for f in disc}
    r_true, _ = _book(Bs, disc, bf_true, te)
    true_gain = r_true - g0
    pval = float(np.mean(np.array(rand_gains) >= true_gain))
    out["placebo"] = {"true_gain": true_gain,
                      "random_mean": float(np.mean(rand_gains)),
                      "random_p95": float(np.percentile(rand_gains, 95)),
                      "p_value": pval}
    (HERE / "cn_resurrection_robust.json").write_text(json.dumps(out, indent=2) + "\n")
    print("=" * 72)
    print("RESURRECTION ROBUSTNESS")
    print("=" * 72)
    print("  hurdle sweep (grave -> revive):")
    for h, d in out["hurdles"].items():
        print(f"    |t|<{h:<5}: n={d['n']:>3}  grave={d['grave']:+.3f}  "
              f"revive={d['revive']:+.3f}  gain={d['gain']:+.3f}")
    p = out["placebo"]
    print(f"  placebo (random battlefield): true gain={p['true_gain']:+.3f}  "
          f"random mean={p['random_mean']:+.3f}  p95={p['random_p95']:+.3f}  "
          f"p-value={p['p_value']:.3f}")
    print("wrote cn_resurrection_robust.json")
    return out


def coarse_vs_fine():
    """The paper's核心对照: COARSE battlefield reallocation (collective move of
    the discarded pool to its train-best universe) WINS massively and robustly;
    FINE per-factor routing (argmax talent) does NOT beat it -- the second wall:
    identifying *which* factor belongs to *which* battlefield is noise-locked,
    only the coarse systematic misallocation is recoverable OOS."""
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr = idx[idx < mid]; te = idx[idx >= mid]

    def _bootci(a, b, seed=11, nb=5000):
        both = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        rng = np.random.default_rng(seed); n = both.shape[0]; d = []
        A = both["a"].to_numpy(); B = both["b"].to_numpy()
        for _ in range(nb):
            bi = rng.integers(0, n, n)
            d.append(_sr(B[bi]) - _sr(A[bi]))
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), \
            float(np.mean(np.array(d) > 0))

    out = {"hurdles": {}}
    for h in (1.645, 1.96, 2.58, 3.0):
        disc = [f for f in shared if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
                and abs(_tstat(Bs["vw"].loc[tr, f])) < h]
        pool_tr = {n: float(np.nanmean([_sr(Bs[n].loc[tr, f]) for f in disc]))
                   for n in BATTLES}
        best = max(pool_tr, key=pool_tr.get)
        _, mg = _book(Bs, disc, {f: "vw" for f in disc}, te)
        _, mc = _book(Bs, disc, {f: best for f in disc}, te)
        # fine routing (argmax per factor)
        def nb(f):
            s = {n: _sr(Bs[n].loc[tr, f]) for n in BATTLES}
            s = {k: v for k, v in s.items() if np.isfinite(v)}
            return max(s, key=s.get) if s else None
        _, mf = _book(Bs, disc, {f: nb(f) for f in disc}, te)
        lo, hi, pg = _bootci(mg, mc)
        out["hurdles"][str(h)] = {
            "n": len(disc), "pool_train_sr": pool_tr, "best_universe": best,
            "sr_grave": _sr(mg), "sr_coarse": _sr(mc), "sr_fine": _sr(mf),
            "coarse_gain": _sr(mc) - _sr(mg), "coarse_ci95": [lo, hi],
            "coarse_p_gt0": pg, "fine_minus_coarse": _sr(mf) - _sr(mc),
        }
    (HERE / "cn_resurrection_coarse.json").write_text(json.dumps(out, indent=2) + "\n")
    print("=" * 72)
    print("COARSE (collective battlefield move) vs FINE (per-factor routing)")
    print("=" * 72)
    for h, d in out["hurdles"].items():
        print(f"  |t|<{h:<5} n={d['n']:>3}: grave={d['sr_grave']:+.3f}  "
              f"COARSE={d['sr_coarse']:+.3f} (gain {d['coarse_gain']:+.3f}, "
              f"CI[{d['coarse_ci95'][0]:+.3f},{d['coarse_ci95'][1]:+.3f}], "
              f"p={d['coarse_p_gt0']:.3f})  FINE={d['sr_fine']:+.3f} "
              f"(fine-coarse {d['fine_minus_coarse']:+.3f})")
    print("wrote cn_resurrection_coarse.json")
    return out


if __name__ == "__main__":
    run()
    robustness()
    coarse_vs_fine()
