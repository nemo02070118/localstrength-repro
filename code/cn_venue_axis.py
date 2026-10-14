"""VENUE-AXIS DIAGNOSTIC: is tradable-rank=0 a property of the SIZE axis only?

The user's deepest hypothesis: the four venues (vw/nyse/me20/wide) are all
variants of the SIZE axis, so the common tilt u = small-cap beta is collinear
and tradable rank collapses to 0. A venue axis ORTHOGONAL to size (information
environment: volatility, liquidity, coverage) might open a tradable rank.

The strict info-environment venue (re-sorting stocks by vol/liquidity) needs
firm-level data (not available here). But we CAN falsify the hypothesis'
premise with the venues on disk, because they are NOT all pure size:
  - me20  : NYSE size-20 breakpoints        -> SIZE axis
  - vw    : value-weight whole market        -> SIZE-ish (cap weighting)
  - nyse  : NYSE-only universe               -> LISTING / LIQUIDITY axis (not
            pure size: it filters by exchange = microstructure/liquidity)
  - wide  : small-cap equal weight           -> SIZE axis (extreme small)

Decompose each venue's advantage into a SIZE component (projection on the
me20<->wide size contrast) and an ORTHOGONAL residual (the non-size, listing/
liquidity part carried by nyse). Then measure tradable persistence SEPARATELY
for the size axis and the orthogonal axis.

  If the orthogonal (non-size) axis has materially higher persistence than the
  size axis, the user is RIGHT: tradable rank is a property of the SIZE venue
  definition, and an information-environment axis is worth building.
  If the orthogonal axis is also ~0 persistence, rank=0 is axis-invariant.

Writes cn_venue_axis.json.
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
HERE = Path(os.environ.get("LOCALSTRENGTH_RESULTS",
                           str(Path(__file__).resolve().parent.parent / "results")))
ANN = np.sqrt(12.0)

def _sr_series(x):
    x = pd.Series(x).dropna()
    if x.size < 12 or x.std(ddof=1) == 0:
        return np.nan
    return float(ANN * x.mean() / x.std(ddof=1))


def venue_sr_matrix(Bs, disc, venues, months):
    """SR of each factor in each venue over `months`. Shape (N, V)."""
    M = np.full((len(disc), len(venues)), np.nan)
    for i, f in enumerate(disc):
        for j, v in enumerate(venues):
            M[i, j] = _sr_series(Bs[v].loc[months, f])
    return M


def persist(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def axis_decomp(S_tr, S_te, venues):
    """Define axis contrasts from venue SR vectors:
      SIZE axis   = wide - me20   (extreme small minus size-20 breakpoints)
      LISTING axis= nyse - vw     (NYSE-only minus whole-market value-weight)
    These two contrasts are the natural orthogonal-ish coordinates. Measure the
    cross-sectional persistence of each contrast score across factors."""
    idx = {v: k for k, v in enumerate(venues)}
    def contrast(S, a, b):
        return S[:, idx[a]] - S[:, idx[b]]
    size_tr = contrast(S_tr, "wide", "me20")
    size_te = contrast(S_te, "wide", "me20")
    list_tr = contrast(S_tr, "nyse", "vw")
    list_te = contrast(S_te, "nyse", "vw")
    # orthogonalize listing axis against size axis (train) to isolate non-size
    m = np.isfinite(size_tr) & np.isfinite(list_tr)
    if m.sum() > 8 and np.std(size_tr[m]) > 0:
        b = np.cov(size_tr[m], list_tr[m])[0, 1] / np.var(size_tr[m])
    else:
        b = 0.0
    listperp_tr = list_tr - b * size_tr
    listperp_te = list_te - b * size_te
    # correlation between the two axes (how collinear are they?)
    mm = np.isfinite(size_tr) & np.isfinite(list_tr)
    axis_corr = (float(np.corrcoef(size_tr[mm], list_tr[mm])[0, 1])
                 if mm.sum() > 8 else np.nan)
    return {
        "size_axis_persistence": persist(size_tr, size_te),
        "listing_axis_persistence": persist(list_tr, list_te),
        "listing_perp_size_persistence": persist(listperp_tr, listperp_te),
        "axis_correlation_size_vs_listing": axis_corr,
        "size_axis_dispersion_tr": float(np.nanstd(size_tr)),
        "listing_axis_dispersion_tr": float(np.nanstd(list_tr)),
    }
def run():
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr, te = idx[idx < mid], idx[idx >= mid]
    disc = [f for f in shared
            if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < 1.96]
    venues = BATTLES
    S_tr = venue_sr_matrix(Bs, disc, venues, tr)
    S_te = venue_sr_matrix(Bs, disc, venues, te)
    ax = axis_decomp(S_tr, S_te, venues)

    out = {"n_disc": len(disc), "venues": venues, "axes": ax,
           "interpretation": {
               "size_axis": "wide - me20 (pure size)",
               "listing_axis": "nyse - vw (listing/liquidity, non-size)"},
           "gate_nonsize_axis_tradable": bool(
               np.isfinite(ax["listing_perp_size_persistence"])
               and ax["listing_perp_size_persistence"] > 0.5),
           "gate_nonsize_beats_size": bool(
               np.isfinite(ax["listing_perp_size_persistence"])
               and np.isfinite(ax["size_axis_persistence"])
               and ax["listing_perp_size_persistence"]
               > ax["size_axis_persistence"] + 0.15)}
    (HERE / "cn_venue_axis.json").write_text(json.dumps(out, indent=2) + "\n")
    _print(out)
    return out


def _print(o):
    print("=" * 76)
    print("VENUE-AXIS DIAGNOSTIC: is rank=0 a SIZE-axis artifact?")
    print("=" * 76)
    a = o["axes"]
    print(f"  N_disc={o['n_disc']}  venues={o['venues']}")
    print("-" * 76)
    print(f"  SIZE axis   (wide-me20) persistence  = "
          f"{a['size_axis_persistence']:+.3f}  (disp={a['size_axis_dispersion_tr']:.3f})")
    print(f"  LISTING axis(nyse-vw)   persistence  = "
          f"{a['listing_axis_persistence']:+.3f}  (disp={a['listing_axis_dispersion_tr']:.3f})")
    print(f"  LISTING perp SIZE       persistence  = "
          f"{a['listing_perp_size_persistence']:+.3f}  <-- the pure non-size axis")
    print(f"  axis correlation (size vs listing)   = "
          f"{a['axis_correlation_size_vs_listing']:+.3f}")
    print("-" * 76)
    print(f"  GATE non-size axis tradable (>0.5):      "
          f"{o['gate_nonsize_axis_tradable']}")
    print(f"  GATE non-size beats size (+0.15):        "
          f"{o['gate_nonsize_beats_size']}")
    print("wrote cn_venue_axis.json")


if __name__ == "__main__":
    run()
