"""Reproduce the three-act load-bearing numbers of "Misallocated, Not Mispriced"
from the shipped result JSONs only -- no raw data and no proprietary code needed.

The frozen result JSONs in results/ are produced from E:/BOAT-FX-DATA by the
scripts in code/ (see run_all.py). This script does NOT recompute them; it reads
them back and prints the exact load-bearing numbers that appear in the paper,
grouped by the paper's three acts, so a third party can confirm that every number
in the manuscript traces to a committed artifact. The arithmetic that combines
them (ratios, fractions) is recomputed here in the open.

  Act I   -- The money:     coarse reallocation revives the discarded (+0.45 SR),
                            and the pure venue spread survives FF6+BAB and GRS.
  Act II  -- Two mirages:   the oracle ceiling is the winner's curse, and fancy
                            weighting is a betting-against-beta relabel.
  Act III -- The theorem:   geometric rank is high but tradable rank is one;
                            plus the economic mechanism and cross-asset reach.

Run:  python reproduce_acts.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def L(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
        return json.load(fh)


def rule(title):
    print("=" * 74)
    print(title)
    print("=" * 74)


def act1():
    rule("ACT I -- THE MONEY: coarse reallocation revives the discarded")
    ca = L("cn_core_alpha.json")
    gr = L("cn_grs.json")
    print(f"  discarded factors (|t|<1.96 on current practice): N = {ca['n_disc']}")
    print(f"  grave  (discard, current practice) SR = {ca['sr_grave']:.3f}")
    print(f"  coarse (reallocated to home venue)  SR = {ca['sr_coarse']:.3f}")
    gain = ca["sr_coarse"] - ca["sr_grave"]
    print(f"  reallocation Sharpe gain over practice = {gain:+.3f}")
    cl = ca["coarse_alpha"]
    print("  coarse-book alpha ladder:")
    for k in ("M0_CAPM", "M1_FF3", "M2_F6", "M3_F6_BAB"):
        if k in cl:
            print(f"      {k:12s} alpha = {cl[k]['alpha_ann_pct']:+.2f}%/yr  "
                  f"t = {cl[k]['t']:.2f}")
    sa = gr["spread_alpha_F6"]
    print(f"  pure venue SPREAD alpha vs FF6 = {sa['alpha_ann_pct']:+.2f}%/yr "
          f"(t = {sa['t']:.2f})")
    g = gr["grs_coarse_plus_spread_vs_F6"]
    print(f"  GRS test (coarse, spread) vs FF6: F({g['N']},{g['T']}) = "
          f"{g['grs_F']:.2f}  p = {g['p_value']:.1e}")
    print("  -> World A: coarse is the deployable capacity. The money is in the "
          "one WHERE decision, made once for the whole pool.")

def act2():
    rule("ACT II -- TWO MIRAGES: fine routing and fancy weighting")
    og = L("cn_oracle_gap.json")
    print("  (1) The oracle ceiling is the winner's curse:")
    print(f"      coarse SR                       = {og['sr_coarse']:.3f}")
    print(f"      oracle, look-ahead (ex post)    = {og['sr_oracle_lookahead']:.3f}")
    print(f"      oracle, out-of-sample pick      = {og['sr_oracle_oos_trainpick']:.3f}")
    gap = og["gap_total_lookahead"]
    nullm = og["gap_null_selection_noise_mean"]
    lo, hi = og["gap_null_ci95"]
    print(f"      look-ahead gap over coarse      = {gap:+.3f}")
    print(f"      null-economy selection bias     = {nullm:+.3f}  CI [{lo:.2f}, {hi:.2f}]")
    print(f"      selection bias / look-ahead gap = {nullm / gap * 100:.0f}%  "
          f"(mirage is {nullm / gap:.1f}x the whole gap)")
    print(f"      genuine tradable signal         = {og['gap_genuine_signal']:+.3f}")
    ec = L("cn_economics.json")
    print("  (2) Fancy weighting is a betting-against-beta relabel:")
    print(f"      {'scheme':5s} {'gross SR':>9s} {'alpha vs 6F+BAB':>16s} {'t':>7s}")
    for s in ("EW", "RP", "MV", "HRP"):
        if s in ec["schemes"]:
            d = ec["schemes"][s]
            a = d.get("alpha_vs_6F_plus_BAB", {})
            at = a.get("alpha_ann_pct", float("nan"))
            tt = a.get("t", float("nan"))
            tag = "  (reference)" if s == "EW" else ""
            print(f"      {s:5s} {d['sharpe_gross']:9.2f} {at:16.2f} {tt:7.2f}{tag}")
    print("  -> Both mirages: the ceiling needs look-ahead, and the super-Sharpe "
          "weighting is low-beta exposure, not new alpha.")

def act3():
    rule("ACT III -- THE THEOREM: geometric rank high, tradable rank one")
    mr = L("cn_multirank.json")
    ct = L("cn_construct.json")
    va = L("cn_venue_axis.json")
    g_mr = mr["multirank_train"]["gamma_multirank"]
    print(f"  geometric multi-rank (venue-interaction)  gamma = {g_mr:.3f}  (> 0.30)")
    print(f"  idiosyncratic direction persistence (OOS)       = "
          f"{mr['eps_direction_persistence']:+.3f}")
    print("  tradable rank across four factor constructions:")
    rows = [
        ("venue-interaction", g_mr, ct["C1_venue_interaction"]["rank"]["tradable_rank"]),
        ("venue-pair", ct["C2_venue_pair"]["rank"]["gamma"],
         ct["C2_venue_pair"]["rank"]["tradable_rank"]),
        ("size-orthogonal", ct["C3_venue_orthogonal"]["rank"]["gamma"],
         ct["C3_venue_orthogonal"]["rank"]["tradable_rank"]),
    ]
    for nm, gm, tr in rows:
        print(f"      {nm:20s} geometric gamma = {gm:4.2f}   tradable rank = {tr}")
    print(f"      non-size venue axis tradable?  {va['gate_nonsize_axis_tradable']} "
          f"({va['interpretation']['listing_axis']})")
    print("  -> Geometric rank in [0.38, 0.78], but tradable rank is 1 or 0 "
          "everywhere. The one tradable direction is the common small-cap tilt.")
    print()
    me = L("cn_mechanism.json")
    rd = me["risk_decomposition"]
    print("  Economic mechanism (what the spread is paid for):")
    print(f"      spread alpha vs FF6              = {rd['F6']['alpha_ann_pct']:+.2f}%/yr "
          f"(t = {rd['F6']['t_alpha']:.2f})")
    print(f"      spread alpha vs FF6 + all legs   = {rd['F6+all']['alpha_ann_pct']:+.2f}%/yr "
          f"(t = {rd['F6+all']['t_alpha']:.2f})")
    cap = me["capacity"]
    net = {int(r["aum_usd"]): r["sr_net"] for r in cap["curve"]}
    b1 = net.get(1000000000)
    b10 = net.get(10000000000)
    print(f"      capacity: gross SR {cap['sr_gross']:.2f}; net SR at $1bn = "
          f"{b1:.3f}, at $10bn = {b10:.3f}")
    print("      cross-asset dispersion vs intermediary capital (HKM):")
    for cls in ("CDS", "USbond", "Sov", "FX", "Commod"):
        if cls in me["cross_asset_hkm"]:
            v = me["cross_asset_hkm"][cls]
            print(f"          {cls:7s} corr = {v.get('corr_intermediary', float('nan')):+.2f}"
                  f"   dispersion SR = {v.get('disp_sharpe', float('nan')):.2f}")
    print("  -> Liquidity-and-reversal premium with a residual the risk set does "
          "not span; large capacity; intermediary-priced in credit and rates.")



def main():
    print("Reproducing the three-act load-bearing numbers from shipped JSONs\n"
          "(no raw data, no proprietary code -- see run_all.py to rebuild JSONs)\n")
    act1()
    print()
    act2()
    print()
    act3()
    print("\nEvery number above is read from results/*.json. The combining "
          "arithmetic (gaps, fractions, ratios) is recomputed in the open in "
          "this script. To rebuild the JSONs from raw data, see run_all.py.")


if __name__ == "__main__":
    main()
