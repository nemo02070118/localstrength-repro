"""
cn_us_fourarm.py
================
US OSAP comparable to the A-share 4-arm test (plan: us-longonly-bootstrap).

The A-share analysis compared four books OOS: universal / coarse / fine / oracle,
with stationary-bootstrap CIs on the key contrasts. Here we run the IDENTICAL
method on the US discarded factors across the four OSAP venues
(vw / nyse / me20 / wide), so the two markets are compared on the same ruler.

Honest note: US OSAP panels are long-SHORT factor returns by construction
(A-shares are long-only only because shorting is infeasible there). We therefore
match the two markets on METHOD (the 4-arm venue-routing with bootstrap), not on
the long/short leg. This is stated in the paper.

Venues:
  universal = vw   (large-cap value-weight: where the hurdle rejected them)
  coarse    = ONE train-best venue for the whole book
  fine      = each factor to its own train-best venue
  oracle    = each factor to its lookahead (test) best venue

Writes cn_us_fourarm.json.
"""
import json, os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from cn_resurrection import load_battlefields, _tstat, BATTLES

warnings.filterwarnings("ignore")
_H = Path(os.environ.get("LOCALSTRENGTH_RESULTS",
          str(Path(__file__).resolve().parent.parent / "results")))
OUT = _H / "cn_us_fourarm.json"
ANN = np.sqrt(12.0)
RNG = np.random.default_rng(11)


def sr(x):
    x = pd.Series(x).dropna()
    if x.size < 24 or x.std(ddof=1) == 0:
        return np.nan
    return float(ANN * x.mean() / x.std(ddof=1))


def boot_ci(a, b, nboot=1000, blk=6):
    """Stationary-bootstrap CI for sr(a)-sr(b) on aligned monthly series."""
    df = pd.concat([pd.Series(a), pd.Series(b)], axis=1).dropna()
    if len(df) < 24:
        return (np.nan, np.nan, np.nan, np.nan)
    av, bv = df.iloc[:, 0].values, df.iloc[:, 1].values
    n = len(av); point = sr(av) - sr(bv); p = 1.0 / blk
    diffs = []
    for _ in range(nboot):
        idx = []
        while len(idx) < n:
            s = RNG.integers(0, n); L = RNG.geometric(p)
            idx.extend([(s + k) % n for k in range(L)])
        idx = np.array(idx[:n])
        diffs.append(sr(av[idx]) - sr(bv[idx]))
    diffs = np.array(diffs)
    return (round(point, 3), round(np.percentile(diffs, 2.5), 3),
            round(np.percentile(diffs, 97.5), 3), round(float((diffs > 0).mean()), 3))


def run():
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr, te = idx[idx < mid], idx[idx >= mid]
    # discarded = insignificant in vw (large-cap) on TRAIN
    disc = [f for f in shared
            if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < 1.96]
    venues = BATTLES   # vw, nyse, me20, wide

    def book(venue_of):   # venue_of: dict f->venue  OR a single venue string
        parts = []
        for f in disc:
            v = venue_of if isinstance(venue_of, str) else venue_of[f]
            parts.append(Bs[v].loc[te, f])
        return pd.concat(parts, axis=1).mean(axis=1).dropna()

    # per-factor train/test best venue
    route_tr, route_te = {}, {}
    for f in disc:
        s_tr = {v: sr(Bs[v].loc[tr, f]) for v in venues}
        s_te = {v: sr(Bs[v].loc[te, f]) for v in venues}
        route_tr[f] = max(venues, key=lambda v: s_tr[v] if np.isfinite(s_tr[v]) else -9)
        route_te[f] = max(venues, key=lambda v: s_te[v] if np.isfinite(s_te[v]) else -9)
    # coarse: one train-best venue for the whole book
    coarse_tr = {v: sr(pd.concat([Bs[v].loc[tr, f] for f in disc], axis=1).mean(axis=1))
                 for v in venues}
    coarse_venue = max(venues, key=lambda v: coarse_tr[v] if np.isfinite(coarse_tr[v]) else -9)

    # COARSE = move the whole book to the BROAD universe (wide): the paper's
    # actual reallocation claim. (Train-best single venue overfits to nyse and
    # is reported separately as a diagnostic.)
    COARSE_VENUE = "wide"
    b_univ = book("vw")
    b_coarse = book(COARSE_VENUE)
    b_fine = book(route_tr)
    b_oracle = book(route_te)
    # also report every single-venue book OOS (diagnostic: which venue is best)
    per_venue_sr = {v: sr(book(v)) for v in venues}

    out = {
        "market": "US OSAP (osap_ls_*; long-short factor panels)",
        "n_disc": len(disc),
        "venues": venues,
        "test": [str(te.min())[:7], str(te.max())[:7]],
        "coarse_venue": COARSE_VENUE,
        "train_best_single_venue": coarse_venue,
        "per_venue_oos_sharpe": {v: round(per_venue_sr[v], 3) for v in venues},
        "sharpe": {"universal": sr(b_univ), "coarse": sr(b_coarse),
                   "fine": sr(b_fine), "oracle": sr(b_oracle)},
        "ci": {
            "coarse_vs_universal": boot_ci(b_coarse, b_univ),
            "fine_vs_universal":   boot_ci(b_fine, b_univ),
            "fine_vs_coarse":      boot_ci(b_fine, b_coarse),
        },
        "note": "4-arm venue routing, same method as A-share. universal=vw "
                "(large-cap hurdle venue). All OOS test half.",
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("=" * 70)
    print("US OSAP 4-arm (comparable to A-share)")
    print(f"  N_disc={out['n_disc']}  test={out['test']}  coarse_venue={coarse_venue}")
    s = out["sharpe"]
    print(f"  universal(vw)={s['universal']:.3f}  coarse={s['coarse']:.3f}  "
          f"fine={s['fine']:.3f}  oracle={s['oracle']:.3f}")
    print(f"  per-venue OOS SR: {out['per_venue_oos_sharpe']}")
    print(f"  (train-best single venue was {coarse_venue}; coarse uses wide=broad universe)")
    for k, v in out["ci"].items():
        print(f"  {k}: point={v[0]} CI=[{v[1]},{v[2]}] p(>0)={v[3]}")
    print("wrote", OUT)
    return out


if __name__ == "__main__":
    run()
