# Dynamical Systems Classification Using Euler Characteristic Profiles
![version](https://img.shields.io/badge/ver.-2-blue)

This project investigates the use of **Euler Characteristic Profiles (ECPs)** for comparing and classifying dynamical systems represented by multidimensional vector fields.

The framework supports several classes of dynamical systems, including:

- 2D linear autonomous systems
- Hopf bifurcation systems
- FitzHugh–Nagumo systems
- 3D Lorenz systems

The workflow generates vector fields on regular Cartesian grids, constructs multidimensional filtrations, computes Euler Characteristic Profiles, calculates pairwise distances, performs hierarchical clustering, and produces visualizations of the obtained results.


---

# Features

## Dynamical systems


The framework currently supports:

| System | Dimension | Parameters | Implementation |
|--------|-----------|------------|----------------|
| Linear autonomous systems | 2D | $a, b, c, d$ | src/systems/examples_linear.py |
| Hopf bifurcation | 2D | $\beta$ | src/systems/examples_HB.py |
| FitzHugh–Nagumo | 2D | $B, I, R, t$ | src/systems/examples_FHN.py |
| Lorenz system | 3D | $\sigma, \rho, \beta$ | src/systems/examples_Lorenz.py |

Each system provides predefined parameter examples. The examples can be selected through the project configuration.

The examples are selected through:

```python
SELECTED_EXAMPLES
```

---

# Configuration

Experiment configuration is stored in:

```text
experiment.toml
```

Important configuration parameters include:

- selected dynamical system,
- selected examples,
- grid limits,
- grid resolution,
- enabled filtrations,
- normalization,
- noise level,
- experiment modes,
- visualization options,
- output directory.


---

# Supported filtrations

The framework constructs filtrations from the original vector field and derived differential features.

The available filtrations depend on the dimension of the vector field.

# Normalization
Vector fields can optionally be normalized before filtration. Normalization is controlled by:

```python
NORMALIZATION_MODE
```
Implementation: src/analysis/normalization.py

## 2D vector fields

For a two-dimensional vector field

$$
F(x,y) = (u(x,y),v(x,y)),
$$

the following filtrations are available:

| Filtration | Feature dimension | Description |
|---|---:|---|
| `VECTOR` | 2D | Original vector field \((u,v)\) |
| `VECTOR_DIV` | 3D | Vector field + divergence |
| `VECTOR_CURL` | 3D | Vector field + scalar curl |
| `VECTOR_ANGLE` | 3D | Vector field + vector angle |
| `EIGS` | 4D | Real and imaginary parts of the two Jacobian eigenvalues |

For a 2D vector field, the Jacobian is

$$
J =
\begin{pmatrix}
\frac{\partial u}{\partial x} &
\frac{\partial u}{\partial y} \\
\frac{\partial v}{\partial x} &
\frac{\partial v}{\partial y}
\end{pmatrix}.
$$

The divergence and curl are computed as

$$
\mathrm{div} F =
\frac{\partial u}{\partial x}
+
\frac{\partial v}{\partial y},
$$

and

$$
\mathrm{curl} F =
\frac{\partial v}{\partial x}
-
\frac{\partial u}{\partial y}.
$$

The angle filtration is defined as

$$
\theta = \mathrm{atan2}(v,u).
$$

The eigenvalue filtration contains

$$
(\mathrm{Re}\lambda_1,
 \mathrm{Im}\lambda_1,
 \mathrm{Re}\lambda_2,
 \mathrm{Im}\lambda_2).
$$

---

## 3D vector fields

For a three-dimensional vector field

$$
F(x,y,z)=(u,v,w),
$$

the available filtrations are:

| Filtration | Feature dimension | Description |
|---|---:|---|
| `VECTOR` | 3D | Original vector field |
| `VECTOR_DIV` | 4D | Vector field + scalar divergence |
| `VECTOR_CURL` | 6D | Vector field + 3D curl |
| `EIGS` | 6D | Real and imaginary parts of the three Jacobian eigenvalues |

The angle filtration is not defined for 3D vector fields.

Implementation: src/analysis/field_features.py

The enabled filtrations are controlled in:

```python
ENABLED_FILTRATIONS
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
|--------|------------|
| L1 | cityblock |
| L2 | euclidean |
| L∞ | chebyshev |

Implementation: src/metrics/distance.py

---

# Project structure

```text
project/
│
├── README.md
├── experiment.toml
├── requirements.txt
│
└── src/
    │
    ├── config.py
    ├── main.py
    ├── runner.py
    │
    ├── analysis/
    │   ├── __init__.py
    │   ├── field_features.py
    │   └── normalization.py
    │
    ├── systems/
    │   ├── README_SYSTEMS.md
    │   ├── __init__.py
    │   ├── examples_FHN.py
    │   ├── examples_HB.py
    │   ├── examples_linear.py
    │   └── examples_Lorenz.py
    │
    ├── metrics/
    │   ├── __init__.py
    │   ├── clustering.py
    │   └── distance.py
    │
    │
    ├── topology/
    │   ├── __init__.py
    │   ├── ecp.py
    │   ├── filtration.py
    │   └── pipeline.py
    │
    ├── utils/
    │   ├── __init__.py
    │   ├── config_snapshot.py
    │   ├── experiment.py
    │   ├── io.py
    │   └── noise.py
    │
    └── visualization/
        ├── __init__.py
        ├── dendrogram.py
        ├── ecp_plot.py
        └── phase_portrait.py
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

---

# Output

The framework automatically generates:

## Dendrograms

Hierarchical clustering visualizations for every filtration.

Location:

```text
output/rrrr_mm_dd/dendrograms/
```

---

## Euler Characteristic Profile plots

Generated for all 2D filtrations.

Location:

```text
output/rrrr_mm_dd/ecp_plots/*.jpg
```

---

## Distance matrices

Saved as NumPy arrays.

Location:

```text
output/rrrr_mm_dd/distance_matrices/*.npy
```

---

## Phase portraits

Optional streamplot visualizations.

Location:

```text
output/rrrr_mm_dd/phase_portraits/*.jpg
```

---

## NPZ archives

All computed distance matrices are collected into:

```text
output/rrrr_mm_dd/all_results.npz
```

---

# Workflow

The complete computational workflow is:

1. Select a dynamical system.
2. Select predefined parameter examples.
3. Generate the computational grid.
4. Generate vector fields (optionally normalize the fields).
5. Build the selected filtrations.
6. Compute Euler Characteristic Contributions and construct Euler Characteristic Profiles.
7. Compute pairwise ECP distances and classical distance matrices.
8. Perform hierarchical clustering and generate dendrograms.
9. Optionally generate ECP plots and phase portraits.

---

# Citation

If this framework contributes to published research, please cite:

- Euler Characteristic Profile methodology [paper](https://doi.org/10.1093/gigascience/giad094)

```bibtex
@article{10.1093/gigascience/giad094,
    author = {Dłotko, Paweł and Gurnari, Davide},
    title = {Euler characteristic curves and profiles: a stable shape invariant for big data problems},
    journal = {GigaScience},
    volume = {12},
    pages = {giad094},
    year = {2023},
    doi = {10.1093/gigascience/giad094}
}
```
- this repository

---
