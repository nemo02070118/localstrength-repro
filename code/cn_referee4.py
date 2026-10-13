"""Referee battery IV: second-line robustness (close the remaining TODOs).

(B15) HXZ q5 spanning of the coarse/gain books.
(A5)  Strict 12-month EMBARGO between train and test.
(A9)  EXPANDING-window walk-forward OOS (re-estimate the discard set and the
      native battlefield every 5 years; no single arbitrary split).
(C30) CAPACITY curve: does the gain concentrate only in the most illiquid
      micro-cap battlefield? recompute dropping the most extreme universe.

All books use decimal returns (OSAP percent /100) to match factor data.
Writes cn_referee4.json.
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
from cn_referee import _nw_t

warnings.filterwarnings("ignore")
HERE = Path(os.environ.get("LOCALSTRENGTH_RESULTS", str(Path(__file__).resolve().parent.parent / "results")))
DATA = Path(os.environ.get("LOCALSTRENGTH_DATA", "E:/BOAT-FX-DATA"))
ANN = np.sqrt(12.0)


def _discard_coarse(Bs, shared, tr, te, hurdle=1.96):
    disc = [f for f in shared if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < hurdle]
    pool = {n: np.nanmean([_sr(Bs[n].loc[tr, f]) for f in disc]) for n in BATTLES}
    cb = max(pool, key=pool.get)
    _, mg = _book(Bs, disc, {f: "vw" for f in disc}, te)
    _, mc = _book(Bs, disc, {f: cb for f in disc}, te)
    return disc, cb, mg / 100.0, mc / 100.0


def q5_spanning():
    q = pd.read_csv(DATA / "q5" / "q5_factors.csv")
    q["date"] = pd.to_datetime(q["date"]) + pd.offsets.MonthEnd(0)
    q = q.set_index("date")
    facs = ["R_MKT", "R_ME", "R_IA", "R_ROE", "R_EG"]
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]; tr = idx[idx < mid]; te = idx[idx >= mid]
    disc, cb, mg, mc = _discard_coarse(Bs, shared, tr, te)
    gain = (mc - mg).dropna()
    out = {}
    for name, series in (("coarse", mc), ("gain", gain)):
        j = pd.concat([series.rename("y")], axis=1).join(q[facs], how="inner").dropna()
        b, t = _nw_t(j["y"].to_numpy(), j[facs].to_numpy(), lags=12)
        out[name] = {"alpha_ann": float(b[0] * 12), "alpha_t": float(t[0]),
                     "betas": {f: float(bi) for f, bi in zip(facs, b[1:])}}
    return out


def embargo(gap=12):
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr = idx[idx < mid]
    te = idx[idx >= idx[len(idx) // 2 + gap]]      # skip `gap` months
    disc, cb, mg, mc = _discard_coarse(Bs, shared, tr, te)
    return {"gap_months": gap, "n_disc": len(disc), "coarse_bf": cb,
            "sr_grave": _sr(mg), "sr_coarse": _sr(mc),
            "gain": _sr(mc) - _sr(mg)}


def walk_forward(start_frac=0.4, step_years=5):
    Bs, shared, idx = load_battlefields()
    n = len(idx)
    anchor = int(n * start_frac)
    streams_g, streams_c = [], []
    marks = []
    i = anchor
    while i < n - 12:
        tr = idx[:i]
        te = idx[i:min(i + 12 * step_years, n)]
        if len(te) < 12:
            break
        disc, cb, mg, mc = _discard_coarse(Bs, shared, tr, te)
        streams_g.append(mg); streams_c.append(mc)
        marks.append((str(te.min())[:7], str(te.max())[:7], cb, len(disc)))
        i += 12 * step_years
    MG = pd.concat(streams_g).sort_index()
    MC = pd.concat(streams_c).sort_index()
    gain = (MC - MG).dropna()
    _, tnw = _nw_t(gain.to_numpy(), lags=12)
    return {"sr_grave": _sr(MG), "sr_coarse": _sr(MC),
            "gain_sr": _sr(gain), "gain_nw_t": float(tnw[0]),
            "n_blocks": len(marks), "oos_start": marks[0][0] if marks else None,
            "oos_end": marks[-1][1] if marks else None}


def capacity():
    """Drop the most illiquid micro universe (me20/wide) one at a time; does the
    coarse gain survive on the more liquid battlefields only?"""
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]; tr = idx[idx < mid]; te = idx[idx >= mid]
    disc = [f for f in shared if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < 1.96]
    out = {}
    for drop in ([], ["wide"], ["wide", "me20"]):
        avail = [b for b in BATTLES if b not in drop]
        pool = {n: np.nanmean([_sr(Bs[n].loc[tr, f]) for f in disc]) for n in avail}
        cb = max(pool, key=pool.get)
        _, mg = _book(Bs, disc, {f: "vw" for f in disc}, te)
        _, mc = _book(Bs, disc, {f: cb for f in disc}, te)
        key = "none" if not drop else "+".join(drop)
        out[f"drop_{key}"] = {"universes": avail, "coarse_bf": cb,
                              "sr_grave": _sr(mg / 100), "sr_coarse": _sr(mc / 100),
                              "gain": _sr(mc / 100) - _sr(mg / 100)}
    return out


def run():
    out = {"q5_spanning": q5_spanning(), "embargo_12m": embargo(12),
           "walk_forward": walk_forward(), "capacity": capacity()}
    (HERE / "cn_referee4.json").write_text(json.dumps(out, indent=2) + "\n")
    print("=" * 74)
    print("REFEREE BATTERY IV  (second-line robustness)")
    print("=" * 74)
    q = out["q5_spanning"]
    print("-- B15. HXZ q5 spanning (Newey-West t) --")
    for k, v in q.items():
        print(f"   {k:7s}: alpha={v['alpha_ann']*100:+.2f}%/yr  t={v['alpha_t']:+.2f}")
    e = out["embargo_12m"]
    print(f"-- A5. 12m embargo: grave={e['sr_grave']:+.3f} coarse={e['sr_coarse']:+.3f} "
          f"gain={e['gain']:+.3f} (bf={e['coarse_bf']}, n={e['n_disc']})")
    w = out["walk_forward"]
    print(f"-- A9. walk-forward {w['oos_start']}..{w['oos_end']} ({w['n_blocks']} blocks): "
          f"grave={w['sr_grave']:+.3f} coarse={w['sr_coarse']:+.3f} "
          f"gain SR={w['gain_sr']:+.3f} NW-t={w['gain_nw_t']:+.2f}")
    print("-- C30. capacity (drop illiquid universes) --")
    for k, v in out["capacity"].items():
        print(f"   {k:16s}: coarse_bf={v['coarse_bf']:5s} grave={v['sr_grave']:+.3f} "
              f"coarse={v['sr_coarse']:+.3f} gain={v['gain']:+.3f}")
    print("wrote cn_referee4.json")
    return out


if __name__ == "__main__":
    run()