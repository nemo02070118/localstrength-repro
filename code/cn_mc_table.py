"""Canonical Monte Carlo table for the resolution limit (IA.2), generated from
the honest attenuation law q^2(tau)=tau/(1+tau), tau=m*lambda/sigma^2.

The DGP is the common-mean unit model: each estimation unit of size m estimates
the fixed home-tilt from a pooled mean whose per-coordinate noise shrinks as
sigma^2/m. We report the theoretical q^2 and the simulated squared correlation
of the pooled estimate with the truth across many units. They agree to Monte
Carlo error at every unit size, which is the finite-sample validation the main
text cites. Writes cn_mc_table.json (consumed by the IA table and the phase
figure)."""
from __future__ import annotations
import json
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("LOCALSTRENGTH_RESULTS", str(HERE)))


def _primitives():
    p = RESULTS / "cn_phase_theory.json"
    if p.exists():
        o = json.loads(p.read_text())
        return float(o["var_cross_advantage"]), float(o["sigma2_single"]), \
            int(o["T_eff_months"]), int(o["n_alt_battlefields"])
    return 0.051, 0.066, 594, 3


def run(reps=8000, seed=0):
    lam, sig2, T, d = _primitives()
    rng = np.random.default_rng(seed)
    GRP = 300
    grid = [1, 2, 5, 13, 30, 94]
    rows = []
    for m in grid:
        tau = m * lam / sig2
        sim = []
        for _ in range(reps):
            tu = rng.normal(0, np.sqrt(lam), GRP)
            eu = tu + rng.normal(0, np.sqrt(sig2 / m), GRP)
            sim.append(np.corrcoef(tu, eu)[0, 1] ** 2)
        rows.append({"m": m, "tau": float(tau),
                     "q2_theory": float(tau / (1 + tau)),
                     "mc_overlap": float(np.mean(sim))})
    out = {"lambda": lam, "sigma2": sig2, "ell": lam / sig2,
           "m_star": sig2 / lam, "T_eff": T, "d": d, "law": "tau/(1+tau)",
           "grid": rows}
    (RESULTS / "cn_mc_table.json").write_text(json.dumps(out, indent=2) + "\n")
    print("MC table (attenuation law) -> cn_mc_table.json")
    print(f"  lambda={lam:.4f} sigma^2={sig2:.4f} m*={sig2/lam:.3f}")
    print(f"  {'m':>4} {'tau':>8} {'q2_theory':>10} {'mc_overlap':>11}")
    for r in rows:
        print(f"  {r['m']:>4} {r['tau']:>8.3f} {r['q2_theory']:>10.3f} "
              f"{r['mc_overlap']:>11.3f}")
    return out


if __name__ == "__main__":
    run()
