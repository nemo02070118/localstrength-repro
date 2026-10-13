"""Narrative-number provenance for Section VI.1 of the paper.

Three claims in the body currently have no JSON backing. This script computes
them from the same train/test split the rest of the paper uses and writes
cn_sparsity.json so every sentence is auditable:

  (a) how many of the N discards earn a positive hindsight (test) gain from
      leaving the coarse venue -- "the gain is sparse";
  (b) the share of the total positive oracle gain contributed by the ten
      largest factors -- "the ten largest account for X%";
  (c) the fraction of discards whose train-argmax venue equals their test-argmax
      venue -- "train-best coincides with test-best X% of the time", with the
      random-guess baseline 1/V.
"""
from __future__ import annotations

import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np

from cn_resurrection import load_battlefields, _tstat, _sr, BATTLES

HERE = Path(os.environ.get("LOCALSTRENGTH_RESULTS", str(Path(__file__).resolve().parent.parent / "results")))


def _argmax(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return max(d, key=d.get) if d else None


def run():
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr = idx[idx < mid]
    te = idx[idx >= mid]
    disc = [f for f in shared
            if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < 1.96]
    srtr = {f: {n: _sr(Bs[n].loc[tr, f]) for n in BATTLES} for f in disc}
    srte = {f: {n: _sr(Bs[n].loc[te, f]) for n in BATTLES} for f in disc}
    pool_tr = {n: float(np.nanmean([srtr[f][n] for f in disc])) for n in BATTLES}
    best = max(pool_tr, key=pool_tr.get)

    # (a) + (b): per-factor hindsight gain = test own-argmax venue minus coarse
    gains = []
    for f in disc:
        own = _argmax(srte[f])  # test-best venue (hindsight)
        g = srte[f].get(own, np.nan) - srte[f].get(best, np.nan)
        if np.isfinite(g):
            gains.append(max(g, 0.0))
    gains = np.array(sorted(gains, reverse=True))
    n_pos = int((gains > 0).sum())
    total = float(gains.sum())
    top10 = float(gains[:10].sum() / total) if total > 0 else None

    # (c) venue hit rate: train-argmax venue == test-argmax venue
    hit = 0
    tot = 0
    for f in disc:
        a_tr = _argmax(srtr[f])
        a_te = _argmax(srte[f])
        if a_tr is not None and a_te is not None:
            tot += 1
            hit += int(a_tr == a_te)
    hit_rate = hit / tot if tot else None

    out = {
        "n_disc": len(disc),
        "coarse_venue": best,
        "n_positive_hindsight_gain": n_pos,
        "frac_positive": n_pos / len(disc),
        "top10_share_of_gain": top10,
        "venue_hit_rate_train_test": hit_rate,
        "random_guess_baseline": 1.0 / len(BATTLES),
        "n_hit_evaluable": tot,
    }
    (HERE / "cn_sparsity.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
