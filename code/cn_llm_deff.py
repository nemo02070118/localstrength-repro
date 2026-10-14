"""
cn_llm_deff.py
==============
The era argument (plan: llm-deff-era). Why the resolution problem EXPLODES in the
LLM factor-zoo age.

A universal t-hurdle controls family-wise error by raising the bar with the
number of candidates N: the Bonferroni/extreme-value hurdle scales like
    kappa_naive(N) = sqrt(2 log N).
But an LLM factory generates N highly REDUNDANT factors spanning only d_eff true
dimensions (d_eff << N). The statistically honest hurdle only needs to control
for the effective dimension:
    kappa_honest(d_eff) = sqrt(2 log d_eff).
The gap kappa_naive - kappa_honest is the OVER-HURDLE: real signals with true
t between the two bars are wrongly killed. We show this gap -- and the fraction
of a fixed signal distribution it kills -- grows with N while d_eff stays flat.

Empirical anchor (cn_llm_contrast.json): the local 6960-factor LLM pool has
d_eff = 16.8, so kappa_naive = sqrt(2 log 6960) = 4.21 but kappa_honest =
sqrt(2 log 16.8) = 2.38. Everything with |t| in (2.38, 4.21) is wrongly discarded.

Writes cn_llm_deff.json. Pure simulation + closed form; no market data needed.
"""
import json, math, os
from pathlib import Path
import numpy as np
from scipy.stats import norm

_H = Path(os.environ.get("LOCALSTRENGTH_RESULTS",
          str(Path(__file__).resolve().parent.parent / "results")))
OUT = _H / "cn_llm_deff.json"
RNG = np.random.default_rng(2026)


def kappa(n):
    return math.sqrt(2.0 * math.log(max(n, 2.0)))


def wrongly_killed_fraction(N, d_eff, true_t_mean=3.0, true_t_sd=1.0, n_sig=200000):
    """Fraction of GENUINE signals (true t ~ N(true_t_mean, true_t_sd), truncated
    at >0) that fall between the honest and naive hurdles and are thus killed by
    the naive universal hurdle but not by the honest one."""
    kn, kh = kappa(N), kappa(d_eff)
    t = RNG.normal(true_t_mean, true_t_sd, n_sig)
    t = t[t > 0]
    killed_naive = np.mean(t < kn)
    killed_honest = np.mean(t < kh)
    over = killed_naive - killed_honest    # wrongly killed by over-hurdle
    return {"kappa_naive": kn, "kappa_honest": kh,
            "frac_killed_naive": float(killed_naive),
            "frac_killed_honest": float(killed_honest),
            "frac_wrongly_killed": float(over)}


if __name__ == "__main__":
    print("[1] closed-form hurdle gap vs N at fixed d_eff ...")
    # Realistic LLM factory: d_eff grows SUBLINEARLY with N (redundancy). We model
    # two regimes: (a) d_eff fixed (pure redundancy), (b) d_eff = c*sqrt(N).
    grid = []
    for N in [100, 500, 1000, 5000, 10000, 50000, 100000]:
        row = {"N": N, "kappa_naive": kappa(N)}
        row["d_eff_fixed20"] = {"d_eff": 20, **wrongly_killed_fraction(N, 20)}
        deff_sqrt = max(2.0, 2.0 * math.sqrt(N) / 10.0)   # c*sqrt(N)
        row["d_eff_sqrtN"] = {"d_eff": round(deff_sqrt, 1),
                              **wrongly_killed_fraction(N, deff_sqrt)}
        grid.append(row)
        print(f"    N={N:6d}  kappa_naive={row['kappa_naive']:.2f}  "
              f"[d_eff=20] wrongly_killed={row['d_eff_fixed20']['frac_wrongly_killed']:.3f}")

    print("[2] empirical anchor: local 6960-factor LLM pool ...")
    emp = {}
    try:
        c = json.loads((Path(__file__).resolve().parent / "cn_llm_contrast.json").read_text())
        N_emp = c["n_llm_factors"]; deff_emp = c["d_eff_llm"]
        emp = {"N": N_emp, "d_eff": deff_emp,
               "kappa_naive": kappa(N_emp), "kappa_honest": kappa(deff_emp),
               **wrongly_killed_fraction(N_emp, deff_emp)}
        print(f"    N={N_emp} d_eff={deff_emp:.1f}  kappa_naive={emp['kappa_naive']:.2f} "
              f"kappa_honest={emp['kappa_honest']:.2f}  "
              f"wrongly_killed={emp['frac_wrongly_killed']:.3f}")
    except Exception as e:
        print("    (could not load cn_llm_contrast.json:", e, ")")

    out = {
        "idea": "universal hurdle sqrt(2 log N) over-penalizes a redundant LLM "
                "zoo whose effective dimension d_eff << N; signals with true t "
                "between sqrt(2 log d_eff) and sqrt(2 log N) are wrongly killed, "
                "and that band widens as the factory scales N up.",
        "hurdle_grid_vs_N": grid,
        "empirical_local_llm": emp,
        "takeaway": "The over-hurdle (kappa_naive - kappa_honest) and the fraction "
                    "of genuine signals it kills grow monotonically with the size "
                    "of the factory, holding the true signal space fixed. The LLM "
                    "era mechanically manufactures discarded-but-real factors.",
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT)
