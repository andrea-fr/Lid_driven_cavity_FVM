# FVM-LidDrivenCavity

2D steady incompressible Navier-Stokes solver using the SIMPLE algorithm on structured collocated quadrilateral meshes. Finite Volume Method with Rhie-Chow interpolation and Second Order Upwind convection scheme.

---

## Overview

This project implements a finite volume solver for the **lid-driven cavity** benchmark problem. The cavity is a unit square with three fixed walls and a top wall moving at constant velocity $U = 1$ m/s. The solver is validated against the reference data of Ghia et al. (1982) for Reynolds numbers ranging from 100 to 10000.

The project consists of two independent modules:

- **`mesher.py`** — generates a structured quadrilateral mesh on the unit square, computes cell connectivity, face normals, centroids and areas, and exports the mesh to `mesh_data.npz` and `mesh_data.xlsx`.
- **`lid_driven_cavity.py`** — reads the mesh and solves the incompressible Navier-Stokes equations using the SIMPLE pressure-velocity coupling algorithm.

---

## Numerical Methods

| Feature | Method |
|---|---|
| Discretization | Finite Volume Method (FVM) |
| Pressure-velocity coupling | SIMPLE |
| Convection scheme | Second Order Upwind (SOU) |
| Diffusion scheme | Central differencing |
| Gradient reconstruction | Weighted Least Squares |
| Gradient limiting | Venkatakrishnan limiter |
| Face flux interpolation | Rhie-Chow interpolation |

---

## Results

Velocity profiles along the centerlines of the cavity are compared against the reference data of Ghia et al. (1982).

---

## Requirements

```
numpy
scipy
matplotlib
pandas
xlsxwriter
```

Install with:

```bash
pip install numpy scipy matplotlib pandas xlsxwriter
```

---

## Usage

**1. Generate the mesh:**

```bash
python mesher.py
```

This produces `mesh_data.npz` and `mesh_data.xlsx` in the same directory. Grid spacing can be set in 

```python
nx, ny = 100
```

**2. Run the solver:**

```bash
python lid_driven_cavity.py
```

The Reynolds number and numerical parameters can be set at the top of `lid_driven_cavity.py`:

```python
Re            = 1000
omega_u_start = 0.5
omega_p_start = 0.3
max_iter      = 2000
tol           = 1e-5
```

Results are saved in the `plots/` subdirectory.

---

## References

- U. Ghia, K.N. Ghia, C.T. Shin, *High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method*, Journal of Computational Physics, 48(3):387–411, 1982.
- J.H. Ferziger, M. Perić, R.L. Street, *Computational Methods for Fluid Dynamics*, 4th ed., Springer, 2020.
- H. Jasak, *Error Analysis and Estimation for the Finite Volume Method with Applications to Fluid Flows*, PhD Thesis, Imperial College London, 1996.
- C.M. Rhie, W.L. Chow, *Numerical study of the turbulent flow past an airfoil with trailing edge separation*, AIAA Journal, 21(11):1525–1532, 1983.
