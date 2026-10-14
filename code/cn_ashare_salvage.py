"""
cn_ashare_salvage.py
=====================
AMBITION EXPERIMENT (truly untested): does the "salvage discarded factors by
routing each to its native venue" thesis replicate OUT-OF-SAMPLE in a *second,
independent market* -- the full China A-share universe (5441 stocks, 2008-2026)?

Data: quanta_alpha/daily_pv.h5  (datetime x instrument : OHLCV + adj factor)

Pipeline
--------
1. Load A-share daily panel, keep real stocks, build monthly adjusted returns.
2. Build a market-cap / liquidity proxy (dollar volume) -> 3 size VENUES:
      large / mid / small  (NYSE-breakpoint analogue for A-shares).
3. Construct classic A-share factors from price/volume only:
      REV (1m reversal), VOL (idio/total vol), TURN (turnover), ILLIQ (Amihud),
      MOM (12-1 momentum), MAX (lottery).
4. For each factor x venue: long-short decile return series.
5. "Local strength" map: which venue is each factor strongest in?
6. SALVAGE mechanism: route each factor to its *train-period* best venue,
      evaluate OOS; compare to (a) universal/pooled venue, (b) oracle venue.
7. Write cn_ashare_salvage.json with every number for the paper appendix.

No fabrication: every statistic is computed from the downloaded panel.
"""
import json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import os as _os
H5 = Path(_os.environ.get("LOCALSTRENGTH_ASHARE", r"E:\BOAT-FX-DATA\quanta_alpha")) / "daily_pv.h5"
OUT = Path(__file__).resolve().parent / "cn_ashare_salvage.json"
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------- load
def load_monthly():
    s = pd.HDFStore(H5, "r")
    d = s["/data"]
    s.close()
    inst = d.index.get_level_values("instrument")
    mask = inst.map(lambda x: bool(re.match(r"(SH6|SZ[03])", x)))
    d = d[mask.values].copy()
    d["adj"] = d["$close"] * d["$factor"]
    d["dvol"] = d["$close"] * d["$volume"]
    d = d.sort_index()
    return d

# ---------------------------------------------------------------- monthly build
def build_monthly(d):
    """Return monthly frames: adj-return R, dollar-volume DV, and daily-vol/turn
    characteristics aggregated to month-end. Index: date x instrument."""
    d = d.copy()
    d["ym"] = d.index.get_level_values("datetime").to_period("M")
    d["inst"] = d.index.get_level_values("instrument")
    # daily simple return per stock (from adj close)
    d["r1d"] = d.groupby("inst")["adj"].pct_change()
    # daily Amihud illiquidity contribution |r|/dollar-vol (vectorized)
    d["illiq_d"] = d["r1d"].abs() / d["dvol"].replace(0, np.nan)
    g = d.groupby(["ym", "inst"])
    # vectorized built-in aggregations only (fast over ~1M groups)
    m = g.agg(
        adj_last=("adj", "last"),
        adj_first=("adj", "first"),
        dvol=("dvol", "mean"),
        vol_d=("r1d", "std"),      # total daily vol in month -> VOL factor
        max_d=("r1d", "max"),      # lottery MAX factor
        amihud=("illiq_d", "mean"),
        ndays=("r1d", "count"),
    )
    # monthly return from month-end adj vs month-start adj (robust vs compounding)
    m["ret_m"] = m["adj_last"] / m["adj_first"] - 1.0
    return m

def ls_return(sig, R, venue_mask=None, q=0.2, tcost=0.0, long_only=True):
    """Monthly factor return series. sig_t predicts R_{t+1}.
    long_only=True : top-quintile LONG portfolio minus the venue EW benchmark
                     (A-shares cannot be shorted; this is the tradable spec).
    long_only=False: top-quintile minus bottom-quintile (academic LS).
    venue_mask restricts the investable set to a size venue.
    tcost: one-way cost (fraction) charged on monthly turnover of the long leg."""
    s = sig.shift(1)
    r = R
    if venue_mask is not None:
        s = s.where(venue_mask)
        r = r.where(venue_mask)
    out = []
    idx = r.index
    pl = set()
    for dt in idx:
        row_s = s.loc[dt].dropna()
        row_r = r.loc[dt].dropna()
        row_s = row_s[row_s.index.isin(row_r.index)]
        if len(row_s) < 20:
            out.append(np.nan); pl = set(); continue
        hi = row_s.quantile(1 - q)
        longs = set(row_s[row_s >= hi].index)
        rl = row_r[list(longs)].mean()
        if long_only:
            bench = row_r[list(row_s.index)].mean()   # venue EW benchmark
            gross = rl - bench
            turn = len(longs ^ pl) / max(len(longs | pl), 1)
            gross -= tcost * turn
            pl = longs
        else:
            lo = row_s.quantile(q)
            shorts = set(row_s[row_s <= lo].index)
            rs = row_r[list(shorts)].mean()
            gross = rl - rs
            tl = len(longs ^ pl) / max(len(longs | pl), 1)
            gross -= tcost * tl
            pl = longs
        out.append(gross)
    return pd.Series(out, index=idx)


def venue_masks(DV, nbins=3):
    """Classify each stock-month into a size VENUE by cross-sectional dollar-vol
    rank: 0=small, 1=mid, 2=large. Returns dict name->bool frame aligned to DV."""
    rank = DV.rank(axis=1, pct=True)
    masks = {
        "small": rank <= 1/3,
        "mid":  (rank > 1/3) & (rank <= 2/3),
        "large": rank > 2/3,
    }
    return masks


def sharpe(x, ann=12):
    x = pd.Series(x).dropna()
    if x.std() == 0 or len(x) < 6:
        return np.nan
    return float(x.mean() / x.std() * np.sqrt(ann))


def stationary_bootstrap_ci(series_a, series_b, nboot=1000, blk=6, ann=12):
    """CI for Sharpe(a) - Sharpe(b) via stationary (block) bootstrap on the
    aligned monthly difference. Returns (point, lo2.5, hi97.5, p_gt0)."""
    df = pd.concat([series_a, series_b], axis=1).dropna()
    if len(df) < 12:
        return (np.nan, np.nan, np.nan, np.nan)
    a = df.iloc[:, 0].values; b = df.iloc[:, 1].values
    n = len(a)
    point = sharpe(a, ann) - sharpe(b, ann)
    diffs = []
    p = 1.0 / blk
    for _ in range(nboot):
        idx = []
        while len(idx) < n:
            start = RNG.integers(0, n)
            L = RNG.geometric(p)
            idx.extend([(start + k) % n for k in range(L)])
        idx = np.array(idx[:n])
        sa = sharpe(a[idx], ann); sb = sharpe(b[idx], ann)
        diffs.append(sa - sb)
    diffs = np.array(diffs)
    return (round(point, 3), round(np.percentile(diffs, 2.5), 3),
            round(np.percentile(diffs, 97.5), 3),
            round(float((diffs > 0).mean()), 3))


if __name__ == "__main__":
    print("[1] loading A-share daily panel ...")
    d = load_monthly()
    print("    rows:", len(d), "stocks:",
          d.index.get_level_values("instrument").nunique())
    print("[2] building monthly panel (this takes a bit) ...")
    m = build_monthly(d)
    del d
    print("    monthly rows:", len(m))

    print("[3] constructing factor signals ...")
    # pivot to date x instrument wide frames
    R = m["ret_m"].unstack("inst")        # realized monthly return
    DV = m["dvol"].unstack("inst")        # size/liquidity proxy (venue sort var)
    VOLd = m["vol_d"].unstack("inst")
    MAXd = m["max_d"].unstack("inst")
    AMI = m["amihud"].unstack("inst")
    R = R.sort_index(); DV = DV.sort_index()
    VOLd = VOLd.reindex_like(R); MAXd = MAXd.reindex_like(R); AMI = AMI.reindex_like(R)

    # signals known AT month t (predict R at t+1). Sign chosen so high=long leg.
    SIG = {
        "REV":   -R,                       # 1m reversal: short past winners
        "MOM":    R.rolling(12).sum().shift(1),   # 12-1 momentum
        "VOL":   -VOLd,                    # low-vol anomaly: long low vol
        "MAX":   -MAXd,                    # lottery: short high-max
        "TURN":  -DV.pct_change(),         # turnover change proxy (short rising)
        "ILLIQ":  AMI,                     # Amihud: long illiquid
    }
    # SECTION_FACTORS
    sv_venues = ["small", "mid", "large"]
    venues = sv_venues + ["all"]
    facs_ok = [f for f in SIG if f != "ILLIQ"]   # ex micro-cap trap factor

    def run_experiment(floor, tcost, long_only):
        """One full spec: tradability floor, cost, long/short mode.
        Returns (series dict, local-strength map, per-factor rows, combos)."""
        liq_rank = DV.rank(axis=1, pct=True)
        tradable = liq_rank > floor
        Rt = R.where(tradable)
        VEN = venue_masks(DV.where(tradable))
        series = {}
        for fac, sig in SIG.items():
            for ven in venues:
                vm = None if ven == "all" else VEN[ven]
                series[(fac, ven)] = ls_return(sig, Rt, vm, tcost=tcost,
                                               long_only=long_only)
        strength = {fac: {ven: sharpe(series[(fac, ven)]) for ven in venues}
                    for fac in SIG}
        dates = R.index
        split = len(dates) // 2
        tr_idx, te_idx = dates[:split], dates[split:]

        def win(s, idx):
            return pd.Series(series[s]).reindex(idx)

        def combo(parts):
            return pd.concat(parts, axis=1).mean(axis=1)

        # ---- COARSE arm: ONE venue for the whole book (train-chosen) --------
        # Pick the single venue that maximizes the TRAIN combo Sharpe across all
        # ex-ILLIQ factors; apply that one venue to every factor in TEST.
        keep_facs = [f for f in SIG if f in facs_ok]
        coarse_tr = {}
        for v in sv_venues:
            tr_parts = [win((f, v), tr_idx) for f in keep_facs]
            coarse_tr[v] = sharpe(combo(tr_parts))
        coarse_venue = max(sv_venues,
                           key=lambda v: coarse_tr[v] if np.isfinite(coarse_tr[v]) else -9)

        rows, pu, ps_, po, pc = {}, [], [], [], []
        for fac in SIG:
            tr = {v: sharpe(win((fac, v), tr_idx)) for v in sv_venues}
            te = {v: sharpe(win((fac, v), te_idx)) for v in sv_venues}
            b_tr = max(sv_venues, key=lambda v: tr[v] if np.isfinite(tr[v]) else -9)
            b_te = max(sv_venues, key=lambda v: te[v] if np.isfinite(te[v]) else -9)
            rows[fac] = {"train_sharpe": tr, "test_sharpe": te,
                         "route_to": b_tr, "oracle_venue": b_te,
                         "coarse_venue": coarse_venue,
                         "universal_test_sharpe": sharpe(win((fac, "all"), te_idx)),
                         "salvage_test_sharpe": te[b_tr],
                         "coarse_test_sharpe": te[coarse_venue],
                         "oracle_test_sharpe": te[b_te]}
            pu.append(win((fac, "all"), te_idx))
            ps_.append(win((fac, b_tr), te_idx))
            po.append(win((fac, b_te), te_idx))
            pc.append(win((fac, coarse_venue), te_idx))

        keep = [i for i, f in enumerate(SIG) if f in facs_ok]
        def c4(pf, ps2, pc2, po2, idxs=None):
            sel = (lambda L: L) if idxs is None else (lambda L: [L[i] for i in idxs])
            return {"universal": sharpe(combo(sel(pf))),
                    "coarse": sharpe(combo(sel(pc2))),
                    "fine": sharpe(combo(sel(ps2))),
                    "oracle": sharpe(combo(sel(po2)))}
        combos = {
            "all": c4(pu, ps_, pc, po),
            "ex_illiq": c4(pu, ps_, pc, po, keep),
            "coarse_venue": coarse_venue,
        }
        # bootstrap CIs (ex-ILLIQ combo): fine-vs-universal AND fine-vs-coarse
        cu = combo([pu[i] for i in keep])
        cc_ = combo([pc[i] for i in keep])
        cf = combo([ps_[i] for i in keep])
        ci = {
            "fine_vs_universal": stationary_bootstrap_ci(cf, cu),
            "coarse_vs_universal": stationary_bootstrap_ci(cc_, cu),
            "fine_vs_coarse": stationary_bootstrap_ci(cf, cc_),
        }
        return series, strength, rows, combos, ci

    print("[4] BASE spec: long-only, floor=0.30, cost=10bps ...")
    LONG_ONLY = True
    base_series, strength, rows, comb_all, ci_base = run_experiment(0.30, 0.001, LONG_ONLY)
    comb = comb_all["all"]; comb_robust = comb_all["ex_illiq"]
    print("    coarse venue chosen:", comb_all["coarse_venue"])
    print("    combo (all) Sharpe:", {k: round(v, 3) for k, v in comb.items() if k != "coarse_venue"})
    print("    combo (ex-ILLIQ) Sharpe:", {k: round(v, 3) for k, v in comb_robust.items()})
    print(f"    fine-vs-universal CI : {ci_base['fine_vs_universal']}")
    print(f"    coarse-vs-universal CI: {ci_base['coarse_vs_universal']}")
    print(f"    fine-vs-coarse CI    : {ci_base['fine_vs_coarse']}")

    print("[5] ROBUSTNESS sweeps (cost x floor) ...")
    sweep = {}
    for floor in [0.20, 0.30, 0.50]:
        for bps in [0.0005, 0.001, 0.002, 0.003]:
            _, _, _, cc, ci = run_experiment(floor, bps, LONG_ONLY)
            key = f"floor{int(floor*100)}_bps{int(bps*10000)}"
            sweep[key] = {"ex_illiq": {k: round(v, 3) for k, v in cc["ex_illiq"].items()},
                          "coarse_venue": cc["coarse_venue"],
                          "fine_vs_coarse_ci": ci["fine_vs_coarse"],
                          "fine_vs_universal_ci": ci["fine_vs_universal"]}
            e = cc["ex_illiq"]
            print(f"    {key}: univ={e['universal']:.2f} coarse={e['coarse']:.2f} "
                  f"fine={e['fine']:.2f} oracle={e['oracle']:.2f} "
                  f"f-vs-c CI={ci['fine_vs_coarse']}")
    print("[6] writing JSON ...")
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, float):
            return None if not np.isfinite(o) else round(o, 4)
        return o
    result = {
        "market": "China A-share (QuantaAlpha daily_pv.h5)",
        "n_stocks_total": int(R.shape[1]),
        "n_months": int(R.shape[0]),
        "sample": [str(R.index[0]), str(R.index[-1])],
        "train_test_split": {"train": [str(R.index[0]), str(R.index[len(R.index)//2 - 1])],
                               "test": [str(R.index[len(R.index)//2]), str(R.index[-1])]},
        "factors": list(SIG.keys()),
        "venues": venues,
        "local_strength_fullsample_sharpe": clean(strength),
        "salvage_per_factor": clean(rows),
        "spec": "LONG-ONLY (A-shares cannot be shorted): top-quintile minus "
                "venue EW benchmark; base floor=30%% illiquid dropped; cost=10bps/side",
        "coarse_venue": comb_all["coarse_venue"],
        "base_combo_test_sharpe_all": clean(comb),
        "base_combo_test_sharpe_ex_illiq": clean(comb_robust),
        "base_ci_ex_illiq": {
            "fine_vs_universal": {"point": ci_base["fine_vs_universal"][0],
                "lo2.5": ci_base["fine_vs_universal"][1], "hi97.5": ci_base["fine_vs_universal"][2],
                "p_gt0": ci_base["fine_vs_universal"][3]},
            "coarse_vs_universal": {"point": ci_base["coarse_vs_universal"][0],
                "lo2.5": ci_base["coarse_vs_universal"][1], "hi97.5": ci_base["coarse_vs_universal"][2],
                "p_gt0": ci_base["coarse_vs_universal"][3]},
            "fine_vs_coarse": {"point": ci_base["fine_vs_coarse"][0],
                "lo2.5": ci_base["fine_vs_coarse"][1], "hi97.5": ci_base["fine_vs_coarse"][2],
                "p_gt0": ci_base["fine_vs_coarse"][3]},
        },
        "robustness_sweep": clean(sweep),
        "note": "4 arms: universal=all-universe venue; coarse=ONE train-chosen "
                "venue for whole book; fine=each factor to its TRAIN-best venue; "
                "oracle=lookahead best venue. All long-only, OOS test half.",
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("    wrote", OUT)
    # headline
    print("\n==== A-SHARE SALVAGE HEADLINE ====")
    print("  [LONG-ONLY, ex-ILLIQ, net of 10bps cost, tradable universe]")
    print(f"  Universal (no routing)   combo Sharpe : {comb_robust['universal']:.3f}")
    print(f"  Coarse (one venue, book) combo Sharpe : {comb_robust['coarse']:.3f}")
    print(f"  Fine (per-factor route)  combo Sharpe : {comb_robust['fine']:.3f}")
    print(f"  Oracle (lookahead)       combo Sharpe : {comb_robust['oracle']:.3f}")
    print(f"  Fine - Coarse (the key test)          : {comb_robust['fine']-comb_robust['coarse']:+.3f}")
    print(f"  Fine-vs-coarse CI (ex-ILLIQ): {ci_base['fine_vs_coarse']}")
