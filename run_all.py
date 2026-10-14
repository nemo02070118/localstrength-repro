#!/usr/bin/env python3
"""Regenerate every load-bearing result JSON from raw data, then print the audit
ledger. Run from the repository root:

    LOCALSTRENGTH_DATA=/path/to/BOAT-FX-DATA python run_all.py

The raw data is not redistributed here (see results/cn_data_provenance.json and
the README for sources). If LOCALSTRENGTH_DATA is unset the scripts fall back to
E:/BOAT-FX-DATA. The frozen outputs that ship in results/ already reproduce the
audit without any raw data; this script rebuilds them from scratch.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
RESULTS = ROOT / "results"

# Producers must run before consumers: resurrection -> referee -> referee4,
# then the standalone calibration and exhibit scripts, then the audit.
ORDER = [
    "cn_resurrection.py",
    "cn_referee.py",
    "cn_referee4.py",
    "cn_phase_theory.py",
    "cn_capacity.py",
    "cn_exhibits.py",
    "cn_sparsity.py",
    "cn_placebo_fdr.py",
    # three-act additions (depend on cn_resurrection, run after it):
    "cn_core_alpha.py",
    "cn_grs.py",
    "cn_oracle_gap.py",
    "cn_multirank.py",
    "cn_construct.py",
    "cn_venue_axis.py",
    "cn_economics.py",
    "cn_mechanism.py",
    # resolution-limit additions (US-side; no large external panel needed):
    "cn_us_fourarm.py",      # 4-arm venue routing, comparable to A-share
    "cn_headtohead_venue.py",# salvage vs fund-proxy spanning
    "cn_llm_deff.py",        # LLM over-hurdle vs effective dimension
]

# Data-gated producers: require the China A-share daily panel (QuantaAlpha
# daily_pv.h5, ~400MB, not redistributed). Run only if LOCALSTRENGTH_ASHARE is
# set to the directory holding daily_pv.h5. Frozen outputs ship in results/.
ASHARE_ORDER = [
    "cn_ashare_salvage.py",
    "cn_ashare_rigor.py",
    "cn_walkforward.py",
    "cn_law_predicts.py",
    "cn_ashare_mechanism.py",
    "cn_llm_salvage.py",
]


def main() -> int:
    if "LOCALSTRENGTH_DATA" not in os.environ:
        print("[warn] LOCALSTRENGTH_DATA not set; scripts will try E:/BOAT-FX-DATA")
    # Scripts write JSON next to themselves; run them with CWD = results so the
    # outputs land in results/ and the audit can read them in place.
    sys.path.insert(0, str(CODE))
    os.chdir(RESULTS)
    for name in ORDER:
        script = CODE / name
        print(f"\n=== running {name} ===", flush=True)
        # Each script resolves its own output path from __file__, so copy it in.
        runpy.run_path(str(script), run_name="__main__")
    if "LOCALSTRENGTH_ASHARE" in os.environ:
        for name in ASHARE_ORDER:
            print(f"\n=== running {name} (A-share, data-gated) ===", flush=True)
            runpy.run_path(str(CODE / name), run_name="__main__")
    else:
        print("\n[info] LOCALSTRENGTH_ASHARE unset; skipping A-share producers. "
              "Frozen A-share outputs already present in results/.", flush=True)
    print("\n=== audit ===", flush=True)
    runpy.run_path(str(CODE / "cn_audit.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
