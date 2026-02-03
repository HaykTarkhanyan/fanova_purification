# Purifying Interaction Effects with the Functional ANOVA

A reproducibility and critical evaluation project for **Lengerich et al. (AISTATS 2020)**.

## Overview

Additive models with interactions (like GA²Ms) have an **identifiability problem**: effects can be freely shifted between main effects and interactions without changing model predictions. This allows "contradictory" interpretations of the same model.

**Purification** solves this by enforcing fANOVA integrate-to-zero constraints via a mass-moving algorithm. The result is a unique, canonical decomposition where each interaction contains only variance that *cannot* be explained by any subset of its features.

This repository implements:
- **Mass-moving purification** for discrete tensors of arbitrary order
- **XGBoost depth≤2 decomposition** into intercept, main effects, and pairwise interactions
- **Experiments** comparing purification under different weight distributions

## Requirements

- Python ≥3.10
- Install dependencies: `pip install -r requirements.txt`

## How to Run

```bash
# Run all experiments (outputs to ./results/)
python run_all_experiments.py
```

Individual experiments can be run directly:
```python
from purify_fanova.experiments import xgb_friedman
xgb_friedman.run(out_dir="results")
```

## Project Structure

```
├── src/                        # Core library (imported as purify_fanova)
│   ├── purify.py               # Mass-moving purification algorithm
│   ├── binning.py              # Feature discretization & weight estimation
│   ├── xgb_decompose.py        # XGBoost → tensor decomposition
│   ├── metrics.py              # Convergence & comparison metrics
│   └── experiments/            # Runnable experiment scripts
│       ├── convergence.py      # Convergence verification
│       ├── boolean_example.py  # Non-identifiability demo
│       ├── xgb_friedman.py     # Friedman #1 dataset experiment
│       ├── correlation_weights.py
│       ├── german_credit.py    # Real-data experiment
│       └── stress_dependence.py
├── data/                       # Input datasets
├── results/                    # Generated outputs (CSVs, PDFs)
├── seminar_specific/           # LaTeX report & slides
└── run_all_experiments.py      # Main entry point
```

## Sources

- **This repository**: [HaykTarkhanyan/fanova_purification](https://github.com/HaykTarkhanyan/fanova_purification)
- **Paper**: [Lengerich et al. (2020) — Purifying Interaction Effects with the Functional ANOVA](https://proceedings.mlr.press/v108/lengerich20a.html), AISTATS 2020, PMLR 108:2402-2412
- **Paper Implementation reference repo**: [AdaptInfer/gam_purification](https://github.com/AdaptInfer/gam_purification)
- **InterpretML**: [interpretml/interpret](https://github.com/interpretml/interpret) — Microsoft's interpretable ML library with GA²M/EBM implementations

## AI Usage Disclosure

The following AI assistants were used to support development of this project:

- **NotebookLM** (Google) — understanding the paper
- **ChatGPT 5.2 Pro** — code generation, documentation
- **Claude Opus / Sonnet 4.5** via **GitHub Copilot** — code refinement, slice generation

All AI-generated content was reviewed, validated, and edited by the author.


### Internal Comment
Repo migrated from https://github.com/HaykTarkhanyan/LMU_Masters/tree/main/05_seminar on 03.02.2026