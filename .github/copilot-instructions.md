# Copilot Instructions for gam_purification

## Project Overview
Reproducibility project for [Lengerich et al. (AISTATS 2020)](https://proceedings.mlr.press/v108/lengerich20a.html) — "Purifying Interaction Effects with the Functional ANOVA".

**The core problem**: Additive models with interactions (like GA²Ms) have an identifiability issue — effects can be freely moved between main effects and interactions without changing predictions. This allows "contradictory" interpretations of the same model.

**The solution**: Mass-moving purification enforces fANOVA integrate-to-zero constraints, yielding a unique canonical decomposition where interactions contain only variance that *cannot* be explained by any subset of features.

## Requirements
- **Python ≥3.10**
- Dependencies: see [requirements.txt](requirements.txt)

## Architecture

### Core Pipeline
1. **XGBoost depth≤2 → Tensors** ([src/xgb_decompose.py](src/xgb_decompose.py)): Decomposes trained models into `T[()]` (intercept), `T[(i,)]` (main effects), `T[(i,j)]` (pairwise interactions)
2. **Binning** ([src/binning.py](src/binning.py)): Discretizes features using tree split thresholds; computes marginal weights
3. **Purification** ([src/purify.py](src/purify.py)): Iterative mass-moving algorithm that shifts slice means from higher-order to lower-order tensors until fANOVA constraints satisfied

### Key Data Structures
```python
Key = Tuple[int, ...]  # e.g., (), (0,), (1,), (0,1)
tensors: Dict[Key, np.ndarray | float]  # Main data structure
weights: Dict[Key, np.ndarray]  # Marginal distributions for each subset
```

### Module Import Pattern
The package is named `purify_fanova` but lives in `src/`. All experiments import as:
```python
from purify_fanova.xgb_decompose import decompose_xgb_depth2, predict_from_tensors
from purify_fanova.binning import discretize_matrix, compute_marginal_weights
from purify_fanova.purify import PurifyConfig, purify_all
```

## Running Experiments

```bash
python run_all_experiments.py  # Runs all 7 experiments, outputs to ./results/
```

Individual experiments are in `src/experiments/` and follow the pattern:
```python
def run(out_dir: Path = Path("results"), **kwargs) -> None:
```

## Key Conventions

### Weight Modes
Experiments compare purification under different weight estimators:
- `"uniform"` — equal bin weights
- `"empirical"` — weights from training data distribution
- `"laplace"` — blend of empirical + uniform (add-one smoothing)
- `"indep"` — product of marginals (independence assumption)

### Tensor Indexing
- Empty tuple `()` is always the intercept (float, not array)
- Single-element tuples `(i,)` are main effects (1D arrays)
- Two-element tuples `(i,j)` with `i < j` are pairwise interactions (2D arrays)

### PurifyConfig
```python
@dataclass(frozen=True)
class PurifyConfig:
    tol: float = 1e-10   # convergence tolerance
    max_iter: int = 200  # max Gauss-Seidel sweeps
```

## Dependencies
Core: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `xgboost` (see [requirements.txt](requirements.txt))

## Testing
**No test suite exists.** Validation is done via sanity checks within experiments (e.g., verifying tensor predictions match XGBoost predictions exactly).

## File Organization
- `src/` — Core library (as `purify_fanova` package)
- `src/experiments/` — Runnable experiment scripts with `run()` functions
- `results/` — Generated CSVs and PDFs (gitignored outputs)
- `data/` — Input datasets (e.g., `german_credit_data.csv`)
- `seminar_specific/` — LaTeX report/slides template

## When Adding Experiments
1. Create `src/experiments/new_experiment.py` with `def run(out_dir: Path, ...) -> None`
2. Add import in `src/experiments/__init__.py`
3. Call from `run_all_experiments.py`
4. Use `compute_marginal_weights()` for weight estimation, `purify_all()` for purification
