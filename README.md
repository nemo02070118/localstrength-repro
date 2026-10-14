# Replication package

**The Resolution Limit of Factor Discovery: Why the Replication Crisis Is a Measurement Problem**
Huannian Jin

This repository reproduces every load-bearing number, table, and figure in the
paper and its Internet Appendix. It ships two layers:

1. **Frozen results** (`results/*.json`) — the exact outputs the paper cites.
   You can reproduce the full audit ledger from these alone, with no raw data.
2. **Code** (`code/*.py`) — the scripts that regenerate those JSONs from the raw
   data. Running them end to end reproduces the frozen results bit for bit.

## Headline numbers

| Quantity | Value | Source JSON |
|---|---|---|
| Grave (status-quo discard) Sharpe | 0.849 | `cn_resurrection_coarse.json` |
| Coarse reallocation Sharpe | 1.302 (**+0.45** over grave) | `cn_resurrection_coarse.json` |
| Fine routing Sharpe | 1.122 (below coarse) | `cn_resurrection_coarse.json` |
| Oracle (hindsight ceiling) Sharpe | 1.447 | `cn_capacity.json` |
| Coarse FF6 alpha | +3.78%/yr, t = 8.27 | `cn_referee.json` |
| Coarse HXZ q5 alpha | +3.41%/yr, t = 7.48 | `cn_referee4.json` |
| Breakeven cost | 80.7 bps one-way | `cn_referee.json` |
| GRS F(8,594) | 5.66, p = 6.1e-07 | `cn_exhibits.json` |
| Factors with recoverable local strength (τ>1) | 6 of 47 spread candidates | `cn_capacity.json` |
| Resolution half-recovery scale | m\* = 1.29 factors | `cn_phase_theory.json` |

### Cross-market, head-to-head, and LLM-era evidence (Ledger G)

| Quantity | Value | Source JSON |
|---|---|---|
| A-share coarse reallocation (long-only) | 0.38 → 1.33, **+0.95** | `cn_ashare_salvage.json` |
| A-share coarse-vs-universal bootstrap CI | [0.49, 1.41], p = 0 | `cn_ashare_salvage.json` |
| A-share reallocation alpha vs mkt+canon | +4.01%/yr, t = 7.83 | `cn_ashare_rigor.json` |
| A-share random-venue placebo | z = 6.6, p = 0; FDR 4/5 | `cn_ashare_rigor.json` |
| US fine-vs-coarse (resolution limit) | −0.18, CI [−0.28, −0.09] | `cn_us_fourarm.json` |
| Law forecast (US λ<0 / A-share λ>0) | predicts routability sign | `cn_law_predicts.json` |
| Salvage incremental alpha vs FF6+BAB | +3.62%/yr, t = 11.6 | `cn_headtohead.json` |
| Risk-parity reverse increment vs salvage | −0.15%/yr, t = −2.8 | `cn_headtohead.json` |
| LLM pool killed by universal hurdle | 94.4% (6573/6960), median \|t\| = 0.71 | `cn_llm_salvage.json` |
| LLM over-hurdle wrongly-killed fraction | 0.62 (N=6960, d_eff=17) | `cn_llm_deff.json` |

## Quick start: reproduce the audit from frozen results

No raw data required.

```bash
pip install -r requirements.txt      # numpy, pandas, pyarrow
python code/cn_audit.py              # prints Ledgers A, B, C
```

The ledger it prints is the single source of truth every table and figure in the
paper is reconciled against.

## Full rebuild from raw data

The raw panels are public but large, so they are **not** redistributed here. See
`results/cn_data_provenance.json` for exact sources. Download them, point the
package at the folder, and rebuild:

```bash
export LOCALSTRENGTH_DATA=/path/to/BOAT-FX-DATA   # Windows: set LOCALSTRENGTH_DATA=...
python run_all.py
```

`run_all.py` runs the six producer scripts in dependency order and then the
audit. Outputs are written to `results/` and should match the shipped JSONs.

### Expected raw-data layout under `LOCALSTRENGTH_DATA`

```
osap_ls_<n>.parquet            # Open Source Asset Pricing long-short legs (Chen-Zimmermann 2022)
ken_french/FF5_monthly.csv     # Fama-French 5 factors (Ken French Data Library)
ken_french/Momentum_monthly.csv
q5/q5_factors.csv              # Hou-Xue-Zhang q5 factors (global-q.org)
```

## Code map

| Script | Produces | Reads |
|---|---|---|
| `code/cn_resurrection.py` | `cn_resurrection_coarse.json`, `cn_resurrection.json`, `cn_resurrection_robust.json` | OSAP parquet |
| `code/cn_referee.py` | `cn_referee.json` (FF alphas, costs, bootstrap CI, spanning) | OSAP, Ken French |
| `code/cn_referee4.py` | `cn_referee4.json` (HXZ q5 spanning, embargo, walk-forward) | OSAP, q5 |
| `code/cn_phase_theory.py` | `cn_phase_theory.json` (BBP calibration: m\*, τ, q²) | OSAP |
| `code/cn_capacity.py` | `cn_capacity.json` (converse bound, τ distribution, router tournament) | OSAP |
| `code/cn_exhibits.py` | `cn_exhibits.json` (summary stats, family decomposition, GRS, CER) | OSAP, Ken French, q5 |
| `code/cn_sparsity.py` | `cn_sparsity.json` (Section VI.1 sparsity numbers: 35/94, top-10 share, venue hit rate) | OSAP |
| `code/cn_placebo_fdr.py` | `cn_placebo_fdr.json` (random-venue & shuffled-label placebos, BH/BY FDR screens) | OSAP |
| `code/cn_us_fourarm.py` | `cn_us_fourarm.json` (US universal/coarse/fine/oracle + bootstrap) | OSAP |
| `code/cn_headtohead_venue.py` | `cn_headtohead.json` (salvage vs fund-proxy spanning) | OSAP, Ken French |
| `code/cn_llm_deff.py` | `cn_llm_deff.json` (over-hurdle vs effective dimension) | `cn_llm_contrast.json` |
| `code/cn_ashare_salvage.py` | `cn_ashare_salvage.json` (A-share 4-arm, data-gated) | A-share `daily_pv.h5` |
| `code/cn_ashare_rigor.py` | `cn_ashare_rigor.json` (spanning, placebo, FDR, NW) | A-share `daily_pv.h5` |
| `code/cn_walkforward.py` | `cn_walkforward.json` (annual walk-forward, both markets) | OSAP + A-share |
| `code/cn_law_predicts.py` | `cn_law_predicts.json` (cross-market law forecast) | OSAP + A-share |
| `code/cn_ashare_mechanism.py` | `cn_ashare_mechanism.json` (arb-cost ladder) | A-share `daily_pv.h5` |
| `code/cn_llm_salvage.py` | `cn_llm_salvage.json` (6960-pool diagnostic + formulaic salvage) | LLM jsonl + A-share |
| `code/cn_audit.py` | Prints the authoritative ledger (A–G) | all result JSONs |

Producers must run before consumers: `cn_referee4` imports helpers from
`cn_resurrection` and `cn_referee`. `run_all.py` enforces the order. The US-side
resolution-limit scripts run with no extra data; the A-share producers are
data-gated on the China panel and run only when `LOCALSTRENGTH_ASHARE` points at
the folder holding `daily_pv.h5` (open QuantaAlpha panel). Frozen A-share outputs
ship in `results/` so the audit reproduces without it.

## Paper source

`paper/` holds the LaTeX source (`LocalStrength.tex`, `LocalStrength_IA.tex`),
the bibliography (`refs.bib`), and the figures (`paper/fig*.png`). To build:

```bash
cd paper
pdflatex LocalStrength && bibtex LocalStrength && pdflatex LocalStrength && pdflatex LocalStrength
pdflatex LocalStrength_IA && bibtex LocalStrength_IA && pdflatex LocalStrength_IA && pdflatex LocalStrength_IA
```

Before compiling you can run the structural validator (no TeX engine needed):

```bash
python code/cn_tex_check.py paper
```

It checks brace/environment balance, that every `\ref` resolves to a `\label`,
that every `\cite` key exists in `refs.bib`, and that every `\includegraphics`
file is present.

## What the paper proves

Coarse reallocation — moving the entire pool of statistically discarded factors
to a single home venue — raises the Sharpe ratio from 0.85 to 1.30, a +0.45 gain
over current industry practice, with an FF6 alpha of 3.78%/yr (t = 8.27). That is
the achievability side.

The converse (Theorem 3) shows coarse is not a heuristic but a capacity. Any
causal routing rule's expected gain over coarse is bounded by Σ q²(τ_i) δ_i,
where q²(τ_i) = 0 for every factor below the BBP threshold τ = 1. Only 6 of 94
factors clear that threshold; the remaining 88 are locked by the data-processing
inequality. A tournament of five routers (per-factor argmax, economic-family
priors, trailing timing, an ML characteristic router, and empirical Bayes) is run
out of sample in `cn_capacity.json`: none beats coarse by more than noise. The
proofs are in `paper/LocalStrength_IA.tex`, Section IA.1.

## License

MIT. See `LICENSE`.
