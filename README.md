# CDS simulation for turbulence nonlinear diffusion and decay 

## Overview
This repository contains simulation code for the Kolmogorov-Barenblatt turbulent energy balance equation in the paper **"Nonlinear Diffusion and Decay of an Expanding Turbulent Blob"** (PNAS, 2026).

## Cell Dynamical System (CDS) Method
Cell Dynamical System (CDS) is a split-step scheme for nonlinear PDEs. Each timestep is computed by:
1. an **onsite (local) update** that advances the decay dynamics independently at each grid cell, and  
2. a **coupling (spatial) update** that applies the nonlinear transport using an isotropic discrete Laplacian on the lattice.

## Quickstart
```bash
pip install numpy scipy numba h5py torch
python cds_turbulence_3d_demo.py
```

## Please cite
```bibtex
@article{matsuzawa2025nonlinear,
  title={Nonlinear Diffusion and Decay of an Expanding Turbulent Blob},
  author={Matsuzawa, Takumi and Zhu, Minhui and Goldenfeld, Nigel and Irvine, William},
  journal={arXiv preprint arXiv:2505.22737},
  year={2025}
}
```
