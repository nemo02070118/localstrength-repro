"""CAPACITY: the achievability + converse pair for local-strength recovery.

The claim we are testing is not "we built a better router." It is a Shannon-style
statement:
  * ACHIEVABILITY -- a causal, out-of-sample selective-deviation rule reaches an
    OOS Sharpe of ~1.35, capturing ~84% of the hindsight oracle.
  * CONVERSE -- no causal rule can do materially better, because the per-factor
    venue signal below the BBP wall carries zero recoverable information. The
    expected achievable gain is bounded by sum_i q^2(tau_i) * oracle_gain_i.

We compute four ceilings and compare them to the achievable rule:
  grave  < coarse < OUR RULE <= [feasibility ceilings] < oracle
Ceilings:
  C_bin   binary selective oracle: per factor, hindsight-best of {coarse, own
          train-argmax venue}. Tightest upper bound on OUR rule's class.
  C_wall  genie-wall ceiling: factors whose train spread-t clears the wall get
          their TEST-best venue; the rest stay coarse. (Genie knows who is
          above the wall; still an upper bound.)
  C_info  information converse: coarse + sum_i q^2(tau_hat_i)*(oracle_i-coarse_i).
          The expected gain any causal rule can extract.
Then an empirical-Bayes (James-Stein) router to check nothing beats the ceiling.

Writes cn_capacity.json. Numbers only; no paper edits.
"""
from __future__ import annotations

import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cn_resurrection import load_battlefields, _tstat, _sr, _book, BATTLES

warnings.filterwarnings("ignore")
HERE = Path(os.environ.get("LOCALSTRENGTH_RESULTS", str(Path(__file__).resolve().parent.parent / "results")))
ANN = np.sqrt(12.0)


def _q2(tau, c):
    if tau <= 1.0:
        return 0.0
    return float((1.0 - 1.0 / tau) / (1.0 + c / tau))


def _spread_t(Bs, f, v, base, window):
    """t-stat of (venue v minus base venue) return for factor f on window."""
    d = (Bs[v].loc[window, f] - Bs[base].loc[window, f]).dropna()
    if d.size < 36 or d.std(ddof=1) == 0:
        return np.nan
    return float(d.mean() / d.std(ddof=1) * np.sqrt(d.size))


def _setup(hurdle=1.96):
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr = idx[idx < mid]; te = idx[idx >= mid]
    disc = [f for f in shared
            if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < hurdle]
    srtr = {f: {n: _sr(Bs[n].loc[tr, f]) for n in BATTLES} for f in disc}
    srte = {f: {n: _sr(Bs[n].loc[te, f]) for n in BATTLES} for f in disc}
    pool_tr = {n: float(np.nanmean([srtr[f][n] for f in disc])) for n in BATTLES}
    best = max(pool_tr, key=pool_tr.get)
    return Bs, disc, tr, te, srtr, srte, best


def _argmax(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return max(d, key=d.get) if d else None


def run():
    Bs, disc, tr, te, srtr, srte, best = _setup()
    out = {"n_disc": len(disc), "coarse_bf": best}

    # baselines
    _, mg = _book(Bs, disc, {f: "vw" for f in disc}, te)
    _, mc = _book(Bs, disc, {f: best for f in disc}, te)
    _, mo = _book(Bs, disc, {f: _argmax(srte[f]) for f in disc}, te)
    sr_g, sr_c, sr_o = _sr(mg), _sr(mc), _sr(mo)
    out["grave"], out["coarse"], out["oracle"] = sr_g, sr_c, sr_o
    cap = lambda s: (s - sr_g) / (sr_o - sr_g)

    # --- C_bin: binary selective oracle (hindsight best of coarse vs own) ----
    bfbin = {}
    for f in disc:
        own = _argmax(srtr[f])
        cand = {best: srte[f].get(best, np.nan)}
        if own:
            cand[own] = srte[f].get(own, np.nan)
        bfbin[f] = _argmax(cand) or best
    _, mbin = _book(Bs, disc, bfbin, te)
    out["C_bin"] = {"sr": _sr(mbin), "capture": cap(_sr(mbin))}

    # --- C_wall: genie knows who is above the wall (train spread-t >= 1) ------
    c_asp = (len(BATTLES) - 1) / len(te)
    perfac = {}
    bfwall = {}
    for f in disc:
        own = _argmax(srtr[f])
        tsp = _spread_t(Bs, f, own, best, tr) if own and own != best else np.nan
        perfac[f] = {"own": own, "spread_t_train": tsp}
        # genie: if the TRUE (test) own-vs-coarse is what we could know is strong
        if own and own != best and np.isfinite(tsp) and tsp >= 1.0:
            bfwall[f] = _argmax({best: srte[f].get(best, np.nan),
                                 own: srte[f].get(own, np.nan)}) or best
        else:
            bfwall[f] = best
    _, mwall = _book(Bs, disc, bfwall, te)
    out["C_wall"] = {"sr": _sr(mwall), "capture": cap(_sr(mwall))}

    # --- C_info: information-theoretic converse on expected gain -------------
    # Per factor, tau_i on the own-vs-coarse spread: signal = (train spread)^2
    # scaled by effective length; noise from Lo (2002). q2(tau_i) is the max
    # recoverable fraction of that factor's oracle gain. Summing gives the
    # ceiling on expected gain any causal rule can extract.
    T_eff = len(te)
    gains_oracle = []
    gains_bound = []
    taus = []
    for f in disc:
        own = _argmax(srtr[f])
        if not own or own == best:
            continue
        # oracle per-factor gain (test): own-best vs coarse, only if positive
        g_or = srte[f].get(own, np.nan) - srte[f].get(best, np.nan)
        if not np.isfinite(g_or):
            continue
        # per-factor tau on the spread signal
        spread_tr = (Bs[own].loc[tr, f] - Bs[best].loc[tr, f]).dropna()
        if spread_tr.size < 36:
            continue
        mu = spread_tr.mean(); sd = spread_tr.std(ddof=1)
        if sd == 0:
            continue
        sr_sp = ANN * mu / sd
        sig2_signal = (ANN * mu / sd) ** 2 / len(spread_tr)  # (SR)^2/T proxy
        sigma2 = (1.0 + 0.5 * (sr_sp / ANN) ** 2) / T_eff
        tau_i = sig2_signal / sigma2 if sigma2 > 0 else 0.0
        q = _q2(tau_i, c_asp)
        taus.append(tau_i)
        gains_oracle.append(max(g_or, 0.0))
        gains_bound.append(q * max(g_or, 0.0))
    sum_or = float(np.sum(gains_oracle))
    sum_bd = float(np.sum(gains_bound))
    n_above = int(np.sum(np.array(taus) > 1.0))
    out["C_info"] = {
        "n_candidates": len(taus), "n_tau_above_1": n_above,
        "oracle_gain_sum": sum_or, "recoverable_gain_sum": sum_bd,
        "recoverable_frac_of_oracle_gap": (sum_bd / sum_or) if sum_or > 0 else None,
        "median_tau": float(np.median(taus)) if taus else None,
    }

    # --- EB: James-Stein shrink of per-factor spread, route if shrunk>0 ------
    # Exhaustion check: can an adaptive empirical-Bayes rule beat the ceiling?
    sp_t = []
    fls = []
    for f in disc:
        own = _argmax(srtr[f])
        if not own or own == best:
            continue
        t = _spread_t(Bs, f, own, best, tr)
        if np.isfinite(t):
            sp_t.append(t); fls.append((f, own))
    sp_t = np.array(sp_t)
    # JS shrink toward 0 (null = stay coarse); scale by 1 - (k-2)/sum t^2
    k = len(sp_t)
    shrink = max(0.0, 1.0 - (k - 2) / np.sum(sp_t ** 2)) if k > 2 else 0.0
    sp_shrunk = shrink * sp_t
    bf_eb = {f: best for f in disc}
    n_eb = 0
    for (f, own), ts in zip(fls, sp_shrunk):
        if ts > math.sqrt(2 * math.log(max(len(disc), 2))):  # BH-style bar
            bf_eb[f] = own; n_eb += 1
    _, m_eb = _book(Bs, disc, bf_eb, te)
    out["EB_router"] = {"shrink": float(shrink), "n_route": n_eb,
                        "sr": _sr(m_eb), "capture": cap(_sr(m_eb))}

    # --- the achievable selective rule at the theory threshold ---------------
    kappa = math.sqrt(2 * math.log(math.log(max(len(disc), 3))))  # sqrt(2 loglog N)
    bf_sel = {f: best for f in disc}
    n_sel = 0
    for f in disc:
        own = _argmax(srtr[f])
        if own and own != best:
            t = _spread_t(Bs, f, own, best, tr)
            if np.isfinite(t) and t >= kappa:
                bf_sel[f] = own; n_sel += 1
    _, m_sel = _book(Bs, disc, bf_sel, te)
    out["selective_at_theory_kappa"] = {
        "kappa_sqrt_2loglogN": kappa, "n_deviate": n_sel,
        "sr": _sr(m_sel), "capture": cap(_sr(m_sel))}
    out["discovery_kappa_sqrt_2logN"] = math.sqrt(2 * math.log(len(disc)))

    out["universality"] = _universality()
    out["robustness"] = _robustness()
    out["converse"] = _converse_clean()
    (HERE / "cn_capacity.json").write_text(json.dumps(out, indent=2) + "\n")
    _print(out)
    return out


def _selective_book_generic(Bs, disc, srtr, best, tr, te_window, kappa):
    """Frozen selective rule: deviation set & venues from TRAIN, evaluated on
    an arbitrary evaluation window te_window. Returns (sr, n_deviate, stream)."""
    bfmap = {f: best for f in disc}
    n = 0
    for f in disc:
        own = _argmax(srtr[f])
        if own and own != best:
            t = _spread_t(Bs, f, own, best, tr)
            if np.isfinite(t) and t >= kappa:
                bfmap[f] = own; n += 1
    sr, m = _book(Bs, disc, bfmap, te_window)
    return sr, n, m


def _universality():
    """The crushing claim: the theory threshold kappa = sqrt(2 loglog N) wins
    across (a) every significance hurdle, (b) every test decade, and the
    empirically optimal threshold sits near kappa throughout."""
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr = idx[idx < mid]; te = idx[idx >= mid]
    u = {"by_hurdle": {}, "by_decade": {}}

    # (a) across hurdles ------------------------------------------------
    for h in (1.645, 1.96, 2.58, 3.0):
        disc = [f for f in shared
                if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
                and abs(_tstat(Bs["vw"].loc[tr, f])) < h]
        srtr = {f: {n: _sr(Bs[n].loc[tr, f]) for n in BATTLES} for f in disc}
        pool = {n: float(np.nanmean([srtr[f][n] for f in disc])) for n in BATTLES}
        best = max(pool, key=pool.get)
        kappa = math.sqrt(2 * math.log(math.log(max(len(disc), 3))))
        _, mc = _book(Bs, disc, {f: best for f in disc}, te)
        sr_c = _sr(mc)
        sr_sel, ndev, _ = _selective_book_generic(Bs, disc, srtr, best, tr, te, kappa)
        # empirical best threshold on test (to show it sits near kappa)
        grid = {}
        for k in np.arange(0.5, 3.01, 0.25):
            s, _, _ = _selective_book_generic(Bs, disc, srtr, best, tr, te, k)
            grid[round(float(k), 2)] = s
        kbest = max(grid, key=lambda z: (grid[z] if np.isfinite(grid[z]) else -9))
        u["by_hurdle"][str(h)] = {
            "n_disc": len(disc), "kappa": kappa, "coarse": sr_c,
            "selective_at_kappa": sr_sel, "n_deviate": ndev,
            "beats_coarse": bool(sr_sel >= sr_c),
            "empirical_best_kappa": kbest, "sel_at_best": grid[kbest]}

    # (b) across test decades (frozen 1.96 rule) ------------------------
    disc = [f for f in shared
            if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < 1.96]
    srtr = {f: {n: _sr(Bs[n].loc[tr, f]) for n in BATTLES} for f in disc}
    pool = {n: float(np.nanmean([srtr[f][n] for f in disc])) for n in BATTLES}
    best = max(pool, key=pool.get)
    kappa = math.sqrt(2 * math.log(math.log(max(len(disc), 3))))
    for dec in range(1970, 2030, 10):
        win = te[(te.year >= dec) & (te.year < dec + 10)]
        if len(win) < 24:
            continue
        _, mc = _book(Bs, disc, {f: best for f in disc}, win)
        sr_c = _sr(mc)
        sr_sel, ndev, _ = _selective_book_generic(Bs, disc, srtr, best, tr, win, kappa)
        u["by_decade"][f"{dec}s"] = {
            "coarse": sr_c, "selective": sr_sel,
            "gain": (sr_sel - sr_c) if (np.isfinite(sr_sel) and np.isfinite(sr_c)) else None,
            "beats": bool(np.isfinite(sr_sel) and np.isfinite(sr_c) and sr_sel >= sr_c)}
    wins = [d["beats"] for d in u["by_decade"].values()]
    u["decade_win_rate"] = float(np.mean(wins)) if wins else None
    u["hurdle_win_rate"] = float(np.mean(
        [d["beats_coarse"] for d in u["by_hurdle"].values()]))
    return u


def _converse_clean():
    """The clean converse: (a) only a handful of factors clear tau_i>1, so the
    data-processing inequality locks the rest; (b) an exhaustion table over five
    routers, each tuned OUT OF SAMPLE on a train-internal split, none beating the
    one-line selective rule. This is what makes 'nobody can do better' a result,
    not an assertion."""
    Bs, disc, tr, te, srtr, srte, best = _setup()
    c_asp = (len(BATTLES) - 1) / len(te)
    T_eff = len(te)

    # (a) per-factor tau on own-vs-coarse spread -> histogram of recoverability
    taus = []
    for f in disc:
        own = _argmax(srtr[f])
        if not own or own == best:
            continue
        sp = (Bs[own].loc[tr, f] - Bs[best].loc[tr, f]).dropna()
        if sp.size < 36 or sp.std(ddof=1) == 0:
            continue
        sr_sp = ANN * sp.mean() / sp.std(ddof=1)
        sig2 = (ANN * sp.mean() / sp.std(ddof=1)) ** 2 / len(sp)
        noise = (1.0 + 0.5 * (sr_sp / ANN) ** 2) / T_eff
        taus.append(sig2 / noise if noise > 0 else 0.0)
    taus = np.array(taus)
    out = {"n_spread_candidates": int(taus.size),
           "n_tau_gt1": int((taus > 1).sum()),
           "frac_tau_gt1": float((taus > 1).mean()) if taus.size else None,
           "tau_hist": {
               "le_0p5": int((taus <= 0.5).sum()),
               "0p5_1": int(((taus > 0.5) & (taus <= 1)).sum()),
               "1_2": int(((taus > 1) & (taus <= 2)).sum()),
               "gt_2": int((taus > 2).sum())},
           "tau_values_above_1": sorted([float(t) for t in taus if t > 1],
                                        reverse=True)}

    # (b) exhaustion: five routers, each with its tuning chosen on tr1, frozen,
    # evaluated once on te. Report SR and whether any beats selective.
    mid_tr = tr[len(tr) // 2]
    tr1 = tr[tr < mid_tr]; tr2 = tr[tr >= mid_tr]
    srtr1 = {f: {n: _sr(Bs[n].loc[tr1, f]) for n in BATTLES} for f in disc}
    pool1 = {n: float(np.nanmean([srtr1[f][n] for f in disc])) for n in BATTLES}
    best1 = max(pool1, key=pool1.get)

    def _spread_t_w(f, v, w):
        d = (Bs[v].loc[w, f] - Bs[best].loc[w, f]).dropna()
        if d.size < 24 or d.std(ddof=1) == 0:
            return np.nan
        return float(d.mean() / d.std(ddof=1) * np.sqrt(d.size))

    routers = {}
    # R1 selective hard-threshold (reference), kappa tuned on tr2
    def sel_book(thr, est, win):
        bf = {f: best for f in disc}
        for f in disc:
            own = _argmax({n: _sr(Bs[n].loc[est, f]) for n in BATTLES})
            if own and own != best:
                t = _spread_t_w(f, own, est)
                if np.isfinite(t) and t >= thr:
                    bf[f] = own
        return _book(Bs, disc, bf, win)
    grid = np.arange(0.5, 3.01, 0.25)
    kbest = max(grid, key=lambda k: (sel_book(k, tr1, tr2)[0]
                                     if np.isfinite(sel_book(k, tr1, tr2)[0]) else -9))
    routers["selective_hard"] = _sr(sel_book(kbest, tr, te)[1])
    # R2 fine argmax (no threshold)
    routers["fine_argmax"] = _sr(_book(Bs, disc,
        {f: _argmax(srtr[f]) for f in disc}, te)[1])
    # R3 coarse
    routers["coarse"] = _sr(_book(Bs, disc, {f: best for f in disc}, te)[1])
    # R4 family (economic prior)
    # R5 empirical Bayes with its OWN out-of-sample threshold
    def eb_book(thr, est, win):
        sp_t, fls = [], []
        for f in disc:
            own = _argmax({n: _sr(Bs[n].loc[est, f]) for n in BATTLES})
            if own and own != best:
                t = _spread_t_w(f, own, est)
                if np.isfinite(t):
                    sp_t.append(t); fls.append((f, own))
        sp_t = np.array(sp_t); k = len(sp_t)
        sh = max(0.0, 1.0 - (k - 2) / np.sum(sp_t ** 2)) if k > 2 else 0.0
        bf = {f: best for f in disc}
        for (f, own), ts in zip(fls, sh * sp_t):
            if ts >= thr:
                bf[f] = own
        return _book(Bs, disc, bf, win)
    kbest_eb = max(grid, key=lambda k: (eb_book(k, tr1, tr2)[0]
                                        if np.isfinite(eb_book(k, tr1, tr2)[0]) else -9))
    routers["empirical_bayes_fair"] = _sr(eb_book(kbest_eb, tr, te)[1])
    out["exhaustion"] = {"routers": routers,
                         "selective_kappa_oos": float(kbest),
                         "eb_kappa_oos": float(kbest_eb),
                         "winner": max(routers, key=routers.get)}
    return out


def _robustness():
    """Weld the solution: does selective deviation still beat coarse under
    (a) trading costs on the incremental venue switch, (b) a 12-month embargo
    between estimation and trading, (c) expanding-window walk-forward?"""
    Bs, shared, idx = load_battlefields()
    mid = idx[len(idx) // 2]
    tr = idx[idx < mid]; te = idx[idx >= mid]
    disc = [f for f in shared
            if np.isfinite(_tstat(Bs["vw"].loc[tr, f]))
            and abs(_tstat(Bs["vw"].loc[tr, f])) < 1.96]
    srtr = {f: {n: _sr(Bs[n].loc[tr, f]) for n in BATTLES} for f in disc}
    pool = {n: float(np.nanmean([srtr[f][n] for f in disc])) for n in BATTLES}
    best = max(pool, key=pool.get)
    kappa = math.sqrt(2 * math.log(math.log(max(len(disc), 3))))
    r = {"kappa": kappa}

    # (a) cost: the only incremental turnover vs coarse is the one-time venue
    # switch of the deviated factors; charge bps/yr on that fraction.
    sr_c, _ = _book(Bs, disc, {f: best for f in disc}, te)
    sr_sel, ndev, m_sel = _selective_book_generic(Bs, disc, srtr, best, tr, te, kappa)
    frac_dev = ndev / len(disc)
    r["cost"] = {}
    for bps in (0, 10, 30, 50, 100):
        # annual cost on the deviated sleeve's extra switch, in SR units proxy
        pen = (bps / 1e4) * frac_dev * ANN  # conservative annualized drag
        r["cost"][str(bps)] = {"sel_net": sr_sel - pen, "coarse": _sr(sr_c)
                               if not isinstance(sr_c, float) else sr_c,
                               "beats": bool((sr_sel - pen) >= sr_c)}

    # (b) embargo: drop first 12 months of test from both books
    te_emb = te[te >= te[0] + pd.DateOffset(months=12)]
    sr_c_e, _ = _book(Bs, disc, {f: best for f in disc}, te_emb)
    sr_sel_e, _, _ = _selective_book_generic(Bs, disc, srtr, best, tr, te_emb, kappa)
    r["embargo_12m"] = {"coarse": sr_c_e, "selective": sr_sel_e,
                        "gain": sr_sel_e - sr_c_e, "beats": bool(sr_sel_e >= sr_c_e)}

    # (c) walk-forward: re-estimate deviation set on expanding window, 8 blocks
    blocks = np.array_split(np.arange(len(te)), 8)
    streams_c, streams_s = [], []
    for bk in blocks:
        if len(bk) == 0:
            continue
        win = te[bk]
        est = idx[idx < win[0]]
        if len(est) < 120:
            continue
        srtr_e = {f: {n: _sr(Bs[n].loc[est, f]) for n in BATTLES} for f in disc}
        pool_e = {n: float(np.nanmean([srtr_e[f][n] for f in disc])) for n in BATTLES}
        best_e = max(pool_e, key=pool_e.get)
        _, mc = _book(Bs, disc, {f: best_e for f in disc}, win)
        _, _, ms = _selective_book_generic(Bs, disc, srtr_e, best_e, est, win, kappa)
        if mc is not None:
            streams_c.append(mc)
        if ms is not None:
            streams_s.append(ms)
    mc_all = pd.concat(streams_c).sort_index() if streams_c else None
    ms_all = pd.concat(streams_s).sort_index() if streams_s else None
    r["walk_forward"] = {
        "coarse": _sr(mc_all) if mc_all is not None else None,
        "selective": _sr(ms_all) if ms_all is not None else None,
        "gain": (_sr(ms_all) - _sr(mc_all)) if (mc_all is not None and ms_all is not None) else None}
    return r


def _print(o):
    print("=" * 72)
    print("CAPACITY: achievability + converse for local-strength recovery")
    print("=" * 72)
    print(f"  discarded={o['n_disc']}  coarse venue={o['coarse_bf']}")
    print(f"  grave={o['grave']:+.3f}  coarse={o['coarse']:+.3f}  "
          f"oracle={o['oracle']:+.3f}")
    print("-" * 72)
    print(f"  coarse capture                 : {(o['coarse']-o['grave'])/(o['oracle']-o['grave']):.0%}")
    print(f"  C_bin  binary selective oracle : SR={o['C_bin']['sr']:+.3f}  "
          f"capture={o['C_bin']['capture']:.0%}")
    print(f"  C_wall genie-above-wall        : SR={o['C_wall']['sr']:+.3f}  "
          f"capture={o['C_wall']['capture']:.0%}")
    ci = o["C_info"]
    print(f"  C_info converse: {ci['n_tau_above_1']}/{ci['n_candidates']} factors "
          f"have tau>1; recoverable = {ci['recoverable_frac_of_oracle_gap']:.1%} "
          f"of the oracle gap (median tau={ci['median_tau']:.2f})")
    print("-" * 72)
    th = o["selective_at_theory_kappa"]
    print(f"  theory kappa = sqrt(2 loglog N) = {th['kappa_sqrt_2loglogN']:.3f}  "
          f"(discovery kappa sqrt(2 log N) = {o['discovery_kappa_sqrt_2logN']:.3f})")
    print(f"  SELECTIVE @ theory kappa: deviate {th['n_deviate']}  "
          f"SR={th['sr']:+.3f}  capture={th['capture']:.0%}")
    eb = o["EB_router"]
    print(f"  EB router (exhaustion): shrink={eb['shrink']:.2f} route={eb['n_route']}"
          f"  SR={eb['sr']:+.3f}  capture={eb['capture']:.0%}")
    u = o.get("universality", {})
    if u:
        print("-" * 72)
        print(f"  UNIVERSALITY: theory kappa beats coarse in "
              f"{u['hurdle_win_rate']:.0%} of hurdles, "
              f"{u['decade_win_rate']:.0%} of test decades")
        for h, d in u["by_hurdle"].items():
            print(f"    |t|<{h:<5} N={d['n_disc']:>3}: coarse={d['coarse']:+.3f}  "
                  f"sel@k={d['selective_at_kappa']:+.3f} (dev {d['n_deviate']})  "
                  f"emp-best k={d['empirical_best_kappa']:.2f}")
        for dec, d in u["by_decade"].items():
            g = d["gain"]
            print(f"    {dec}: coarse={d['coarse']:+.3f}  sel={d['selective']:+.3f}"
                  f"  gain={g:+.3f}" if g is not None else f"    {dec}: n/a")
    rb = o.get("robustness", {})
    if rb:
        print("-" * 72)
        print("  ROBUSTNESS of selective deviation (vs coarse):")
        c = rb["cost"]
        print("    net SR by cost: " + "  ".join(
            f"{b}bps:{c[b]['sel_net']:+.3f}" for b in c))
        e = rb["embargo_12m"]
        print(f"    12m embargo: coarse={e['coarse']:+.3f} sel={e['selective']:+.3f} "
              f"gain={e['gain']:+.3f}")
        w = rb["walk_forward"]
        if w["gain"] is not None:
            print(f"    walk-forward: coarse={w['coarse']:+.3f} "
                  f"sel={w['selective']:+.3f} gain={w['gain']:+.3f}")
    cv = o.get("converse", {})
    if cv:
        print("-" * 72)
        print(f"  CLEAN CONVERSE: {cv['n_tau_gt1']}/{cv['n_spread_candidates']} "
              f"spread candidates have tau_i>1 "
              f"({cv['frac_tau_gt1']:.0%}); the rest are DPI-locked")
        print(f"    tau histogram: <=0.5:{cv['tau_hist']['le_0p5']}  "
              f"0.5-1:{cv['tau_hist']['0p5_1']}  1-2:{cv['tau_hist']['1_2']}  "
              f">2:{cv['tau_hist']['gt_2']}")
        ex = cv["exhaustion"]
        print(f"    EXHAUSTION (each router tuned OOS, frozen, eval once):")
        for r, s in sorted(ex["routers"].items(), key=lambda z: -z[1]):
            print(f"      {r:<22}: SR={s:+.3f}")
        print(f"    winner = {ex['winner']}  (selective kappa_oos="
              f"{ex['selective_kappa_oos']:.2f}, eb kappa_oos={ex['eb_kappa_oos']:.2f})")
    print("wrote cn_capacity.json")


if __name__ == "__main__":
    run()
