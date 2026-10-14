"""Cell-level audit: pull every load-bearing number from the result JSONs so the
paper can be reconciled line by line. Prints an authoritative ledger."""
import json
import os
from pathlib import Path

H = Path(os.environ.get("LOCALSTRENGTH_RESULTS",
                        str(Path(__file__).resolve().parent.parent / "results")))
def L(n): return json.loads((H / n).read_text())

rc = L('cn_resurrection_coarse.json')['hurdles']['1.96']
rf = L('cn_referee.json'); r4 = L('cn_referee4.json')
pt = L('cn_phase_theory.json'); cap = L('cn_capacity.json')
ex = L('cn_exhibits.json')

print("=" * 60)
print("LEDGER A -- GRAVE -> COARSE  (the crush over current practice)")
print("=" * 60)
print(f"grave   SR = {rc['sr_grave']:.3f}")
print(f"coarse  SR = {rc['sr_coarse']:.3f}   gain = {rc['coarse_gain']:.3f}")
print(f"fine    SR = {rc['sr_fine']:.3f}   (fine-coarse {rc['fine_minus_coarse']:.3f})")
print(f"coarse gain 95% CI = {rc['coarse_ci95']}   p>0 = {rc['coarse_p_gt0']}")
ra = rf['risk_adjustment']
for k in ('coarse_CAPM', 'coarse_FF3', 'coarse_FF5', 'coarse_FF6'):
    print(f"{k:12s} alpha = {ra[k]['alpha_ann']*100:+.2f}%/yr  t = {ra[k]['alpha_t']:.2f}")
print(f"gain_FF6     alpha = {ra['gain_FF6']['alpha_ann']*100:+.2f}%/yr  t = {ra['gain_FF6']['alpha_t']:.2f}")
print(f"canon span   alpha = {rf['canon_spanning']['alpha_ann']*100:+.2f}%/yr  t = {rf['canon_spanning']['alpha_t']:.2f}")
q5 = r4['q5_spanning']['coarse']
print(f"HXZ q5       alpha = {q5['alpha_ann']*100:+.2f}%/yr  t = {q5['alpha_t']:.2f}")
print(f"breakeven cost = {rf['breakeven_bps_oneway_gain']:.1f} bps one-way")
print(f"CER gain (gamma=3) = {ex['cer_gain']['3']['cer_gain_pct']:.2f}%/yr")
print(f"GRS F({ex['grs_test']['n_families']},{ex['grs_test']['T']}) = {ex['grs_test']['grs_F']:.2f}  p = {ex['grs_test']['p_value']:.2e}")

print()
print("=" * 60)
print("LEDGER B -- CONVERSE  (coarse already touches capacity)")
print("=" * 60)
print(f"oracle SR = {cap['oracle']:.3f}  (coarse captures "
      f"{(cap['coarse']-cap['grave'])/(cap['oracle']-cap['grave'])*100:.0f}% of grave->oracle)")
cv = cap['converse']
print(f"spread candidates with tau_i>1: {cv['n_tau_gt1']}/{cv['n_spread_candidates']} "
      f"= {cv['frac_tau_gt1']*100:.0f}%")
print(f"tau histogram: {cv['tau_hist']}")
print("EXHAUSTION (each router OOS-tuned on train-internal split, frozen, 1 eval):")
for k, v in sorted(cv['exhaustion']['routers'].items(), key=lambda z: -z[1]):
    print(f"   {k:22s} SR = {v:.3f}")
print(f"   winner = {cv['exhaustion']['winner']}")
print(f"   selective kappa_oos = {cv['exhaustion']['selective_kappa_oos']}  "
      f"eb kappa_oos = {cv['exhaustion']['eb_kappa_oos']}")

print()
print("=" * 60)
print("LEDGER C -- WALL CALIBRATION")
print("=" * 60)
print(f"m* (tau=1) = {pt['m_star_tau1']:.3f}")
print(f"tau(m=1) = {pt['predicted']['single (m=1)']['tau']:.3f}  "
      f"q2 = {pt['predicted']['single (m=1)']['q2']:.3f}")
print(f"tau(pool m=94) = {pt['predicted']['pool (m=N)']['tau']:.1f}  "
      f"q2 = {pt['predicted']['pool (m=N)']['q2']:.3f}")
print(f"var_cross = {pt['var_cross_advantage']:.4f}  sigma2 = {pt['sigma2_single']:.4f}")
print(f"empirical single-factor persist rho = {pt['empirical_single_residual_rho']:.3f}")

sp = L('cn_sparsity.json')
print()
print("=" * 60)
print("LEDGER D -- SPARSITY (Section VI.1 narrative numbers)")
print("=" * 60)
print(f"positive hindsight gain: {sp['n_positive_hindsight_gain']}/{sp['n_disc']} "
      f"= {sp['frac_positive']*100:.0f}%")
print(f"top-10 share of total gain = {sp['top10_share_of_gain']*100:.0f}%")
print(f"venue hit rate (train-best = test-best) = {sp['venue_hit_rate_train_test']*100:.0f}% "
      f"(random baseline {sp['random_guess_baseline']*100:.0f}%)")

pf = L('cn_placebo_fdr.json')
print()
print("=" * 60)
print("LEDGER E -- PLACEBO + FDR (referee robustness)")
print("=" * 60)
pr = pf['placebo_random_venue']
print(f"random-venue placebo: mean gain {pr['mean_gain']:+.3f}  true {pr['true_gain']:+.3f}  "
      f"z={pr['z_true_vs_placebo']:+.2f}  p(placebo>=true)={pr['emp_p_true_ge_placebo']:.3f}")
print(f"shuffled-labels placebo: p(placebo>=true)="
      f"{pf['placebo_shuffled_labels']['emp_p_true_ge_placebo']:.3f}")
for k, v in pf['fdr_screens'].items():
    print(f"  {k:10s}: N={v['n_disc']:>3}  gain={v['gain']:+.3f}")

og = L('cn_oracle_gap.json'); ec = L('cn_economics.json'); me = L('cn_mechanism.json')
print()
print("=" * 60)
print("LEDGER F -- MIRAGES + MECHANISM (Sections 7-10)")
print("=" * 60)
print(f"oracle look-ahead SR = {og['sr_oracle_lookahead']:.3f}  "
      f"OOS train-pick = {og['sr_oracle_oos_trainpick']:.3f}  coarse = {og['sr_coarse']:.3f}")
print(f"look-ahead gap = {og['gap_total_lookahead']:+.3f}  "
      f"null selection bias = {og['gap_null_selection_noise_mean']:+.3f} "
      f"CI {og['gap_null_ci95']}")
print("weighting alpha vs 6F+BAB (small-cap venue):")
for s in ('RP', 'MV', 'HRP'):
    a = ec['schemes'][s]['alpha_vs_6F_plus_BAB']
    print(f"   {s:4s}: alpha = {a['alpha_ann_pct']:+.2f}%/yr  t = {a['t']:+.2f}")
rd = me['risk_decomposition']
print(f"spread alpha: FF6 = {rd['F6']['alpha_ann_pct']:+.2f}% (t={rd['F6']['t_alpha']:.2f})  "
      f"FF6+all legs = {rd['F6+all']['alpha_ann_pct']:+.2f}% (t={rd['F6+all']['t_alpha']:.2f})")
cp = me['capacity']
print(f"capacity: gross SR {cp['sr_gross']:.2f}; net SR at $1bn/$10bn = "
      f"{cp['curve'][4]['sr_net']:.3f}/{cp['curve'][6]['sr_net']:.3f}")
print("cross-asset dispersion intermediary-capital corr:")
for cls in ('CDS', 'USbond', 'Sov', 'FX', 'Commod'):
    if cls in me['cross_asset_hkm']:
        v = me['cross_asset_hkm'][cls]
        print(f"   {cls:7s}: corr = {v.get('corr_intermediary', float('nan')):+.2f}  "
              f"disp SR = {v.get('disp_sharpe', float('nan')):.2f}")

print()
print("=" * 60)
print("LEDGER G -- RESOLUTION LIMIT: CROSS-MARKET + HEAD-TO-HEAD + LLM")
print("=" * 60)
def _opt(n):
    p = H / n
    return json.loads(p.read_text()) if p.exists() else None
uf = _opt('cn_us_fourarm.json')
if uf:
    s = uf['sharpe']; ci = uf['ci']
    print(f"US 4-arm: univ {s['universal']:.3f} -> coarse {s['coarse']:.3f} -> "
          f"fine {s['fine']:.3f} -> oracle {s['oracle']:.3f}")
    print(f"   coarse-vs-univ {ci['coarse_vs_universal'][0]:+.3f} CI"
          f"[{ci['coarse_vs_universal'][1]},{ci['coarse_vs_universal'][2]}];  "
          f"fine-vs-coarse {ci['fine_vs_coarse'][0]:+.3f} CI"
          f"[{ci['fine_vs_coarse'][1]},{ci['fine_vs_coarse'][2]}]")
asv = _opt('cn_ashare_salvage.json'); arg = _opt('cn_ashare_rigor.json')
if asv:
    e = asv['base_combo_test_sharpe_ex_illiq']; c = asv['base_ci_ex_illiq']
    print(f"A-share: univ {e['universal']:.3f} -> coarse {e['coarse']:.3f} -> "
          f"fine {e['fine']:.3f} -> oracle {e['oracle']:.3f}")
    print(f"   coarse-vs-univ {c['coarse_vs_universal']['point']:+.3f} CI"
          f"[{c['coarse_vs_universal']['lo2.5']},{c['coarse_vs_universal']['hi97.5']}]")
if arg:
    sp = arg['R1_spanning']['fine_vs_mkt_canon']; pl = arg['R2_placebo']
    print(f"   A spanning fine vs mkt+canon alpha {sp['alpha_ann_pct']:+.2f}%/yr "
          f"t={sp['t_alpha']:.2f}; placebo z={pl['z_true_vs_placebo']:.2f} "
          f"p={pl['emp_p_true_ge']:.3f}; FDR {arg['R3_fdr']['n_pass']}/{arg['R3_fdr']['n_tests']}")
lp = _opt('cn_law_predicts.json')
if lp:
    for mk in ('US', 'A-share'):
        r = lp['markets'][mk]
        print(f"   law {mk}: lambda={r['lambda']:+.3f} tau={r['tau_m1']:+.3f} "
              f"predicted_capture={r['predicted_capture_frac']} "
              f"realized={r['realized_capture_frac']}")
hh = _opt('cn_headtohead.json')
if hh:
    si = hh['spanning_salvage_increment']; rv = hh['reverse_proxy_increment']
    print(f"   head-to-head: salvage vs FF6+BAB {si['vs_FF6_BAB_only']['alpha_ann_pct']:+.2f}% "
          f"t={si['vs_FF6_BAB_only']['t']:.1f}; vs RP {si['vs_risk_parity']['alpha_ann_pct']:+.2f}% "
          f"t={si['vs_risk_parity']['t']:.1f}; reverse RP {rv['risk_parity_vs_salvage']['alpha_ann_pct']:+.2f}% "
          f"t={rv['risk_parity_vs_salvage']['t']:.1f}")
dl = _opt('cn_llm_deff.json'); lsv = _opt('cn_llm_salvage.json')
if dl:
    em = dl['empirical_local_llm']
    print(f"   LLM d_eff: N={em['N']} d_eff={em['d_eff']:.1f} kappa_naive={em['kappa_naive']:.2f} "
          f"kappa_honest={em['kappa_honest']:.2f} wrongly_killed={em['frac_wrongly_killed']:.3f}")
if lsv:
    pa = lsv['part_a_local_6960']
    print(f"   LLM pool: killed {pa['frac_killed_hurdle_1p96']:.3f} ({pa['n_killed']}) "
          f"median|t|={pa['median_abs_t']:.2f}")
