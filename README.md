# Replication package

**Misallocated, Not Mispriced: The Resolution Limit of Local Strength in the Factor Zoo**
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
| BBP resolution threshold | m\* = 1.29 factors | `cn_phase_theory.json` |

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
| `code/cn_audit.py` | Prints the authoritative ledger | all result JSONs |

Producers must run before consumers: `cn_referee4` imports helpers from
`cn_resurrection` and `cn_referee`. `run_all.py` enforces the order.

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
