# Linear Autonomous Systems Classification Using Euler Characteristic Profiles

This project investigates the use of **Euler Characteristic Profiles (ECPs)** for comparing and classifying two-dimensional linear autonomous dynamical systems.

The workflow reproduces and extends the original experimental pipeline by generating multiple canonical vector fields, computing multidimensional Euler Characteristic Profiles, constructing pairwise distance matrices, and visualizing hierarchical clustering results via dendrograms.

---

# Features

## Dynamical systems

The framework generates a collection of canonical 2D linear autonomous systems:

- Stable nodes
- Unstable nodes
- Saddles
- Stable focuses
- Unstable focuses
- Centers

Each system is represented as:

\[
\frac{dx}{dt}=ax+by
\]

\[
\frac{dy}{dt}=cx+dy
\]

and sampled on a regular Cartesian grid.

---

## Supported filtrations

The following multidimensional filtrations are available:

| Filtration | Dimension | Description |
|------------|------------|-------------|
| VECTOR | 2D | Original vector field \((u,v)\) |
| DIV | 3D | Vector field + divergence |
| CURL | 3D | Vector field + curl |
| ANGLE | 3D | Vector field + vector angle |
| EIGS | 4D | Jacobian eigenvalue representation |

The enabled filtrations are controlled in:

```python
ENABLED_FILTRATIONS
```

inside:

```text
src/config.py
```

---

## Experiment modes

Three experiment modes are supported.

### CLEAN

Only original vector fields.

```python
ExperimentMode.CLEAN
```

### NOISY

Gaussian noise added proportionally to the standard deviation of each field.

```python
ExperimentMode.NOISY
```

### FULL

Runs:

- clean dataset
- noisy dataset
- merged dataset

and computes all corresponding ECP analyses.

```python
ExperimentMode.FULL
```

Enabled modes are defined in:

```python
ENABLED_EXPERIMENT_MODES
```

---

## Distance metrics

### Euler Characteristic Profile distance

Computed using:

```python
difference_ECP()
```

from the `pyEulerCurves` package.

Supports:

- exact ECP comparison
- optional discretization
- arbitrary filtration dimension

### Classical vector-space distances

For comparison purposes the framework also computes:

| Metric | scipy name |
|----------|----------|
| L1 | cityblock |
| L2 | euclidean |
| L∞ | chebyshev |

---

# Project structure

```text
linear-autonomous-ecp/
│
├── README.md
├── requirements.txt
│
├── src/
│   │
│   ├── main.py
│   ├── config.py
│   │
│   ├── analysis/
│   │   ├── field_features.py
│   │   └── normalization.py
│   │
│   ├── generators/
│   │   └── examples.py
│   │
│   ├── metrics/
│   │   ├── clustering.py
│   │   └── distance.py
│   │
│   ├── models/
│   │   └── vector_field.py
│   │
│   ├── topology/
│   │   ├── ecp.py
│   │   ├── experiment_runner.py
│   │   ├── filtration.py
│   │   └── pipeline.py
│   │
│   ├── utils/
│   │   ├── config_snapshot.py
│   │   ├── experiment.py
│   │   ├── io.py
│   │   ├── logger.py
│   │   ├── noise.py
│   │   └── summary.py
│   │
│   └── visualization/
│       ├── dendrogram.py
│       ├── ecp_plot.py
│       └── phase_portrait.py
│   
│
└── tests/
```

---

# Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate:

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running experiments

Execute:

```bash
python -m src.main
```

or

```bash
python src/main.py
```

depending on your environment.

---

# Configuration

All experiment parameters are stored in:

```text
src/config.py
```

Important parameters:

```python
GRID_POINTS = 101

NOISE_FRACTION = 0.01

DEFAULT_RANGE = (-150.0, 150.0)

DEFAULT_RESOLUTION = 101 * 101
```

---

## Enable/disable filtrations

Example:

```python
ENABLED_FILTRATIONS = [

    Filtration.VECTOR,
    Filtration.DIV,
    Filtration.CURL,

]
```

---

## Enable experiment modes

Example:

```python
ENABLED_EXPERIMENT_MODES = [
    ExperimentMode.CLEAN
]
```

---

## Enable phase portrait generation

```python
SAVE_PHASE_PORTRAITS = True
```

---

## Enable ECP image generation

```python
SAVE_ECP_IMAGES = True
```

---

# Output

The framework automatically generates:

## Dendrograms

Hierarchical clustering visualizations for every filtration.

Location:

```text
output/dendrograms/
```

---

## Euler Characteristic Profile plots

Generated for all 2D filtrations.

Location:

```text
output*/rrrr_mm_dd_hh-mm-ss/ecp_plots/
```

---

## Distance matrices

Saved as NumPy arrays.

Location:

```text
output*/rrrr_mm_dd_hh-mm-ss/*/*_distance_matrix.npy
```

---

## Phase portraits

Optional streamplot visualizations.

Location:

```text
output*/rrrr_mm_dd_hh-mm-ss/phase_clean/
output*/rrrr_mm_dd_hh-mm-ss/phase_noisy/
```

---

## NPZ archives

All computed distance matrices are collected into:

```text
output*/rrrr_mm_dd_hh-mm-ss/npz/all_results.npz
```

---

# Workflow

The complete workflow executed by the framework is:

1. Generate canonical linear systems
2. Sample vector fields on a grid
3. Build requested filtrations
4. Compute Euler Characteristic Contributions
5. Optionally discretize contributions
6. Compute pairwise ECP distances
7. Generate distance matrices
8. Perform hierarchical clustering
9. Save dendrograms
10. Save optional ECP plots
11. Save all results to NPZ archives

---

# Reproducibility

Experiments are deterministic due to a fixed random seed:

```python
RANDOM_SEED = 42
```

defined in:

```python
src/main.py
```

---

# Dependencies

Main packages:

- numpy
- scipy
- matplotlib
- pyEulerCurves
- tqdm

Optional development tools:

- pytest
- black
- isort
- ruff
- mypy

---

# Citation

If this framework contributes to published research, please cite:

- Euler Characteristic Profile methodology
- pyEulerCurves package
- this repository

---

# License

MIT License
