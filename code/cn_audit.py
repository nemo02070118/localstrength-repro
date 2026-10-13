"""Cell-level audit: pull every load-bearing number from the result JSONs so the
paper can be reconciled line by line. Prints an authoritative ledger."""
import os
import json
from pathlib import Path

H = Path(os.environ.get("LOCALSTRENGTH_RESULTS", str(Path(__file__).resolve().parent.parent / "results")))
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
