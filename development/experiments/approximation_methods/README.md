# Approximation Methods Comparison

This directory contains research code for comparing different statistical 
approximation methods for likelihood computation.

## Scripts

- `nd_approximation.py` - Compare Edgeworth, Gaussian, and Generalized Laplace 
  approximations against exact characteristic function solutions

## Purpose

Validates which approximation methods work best for different parameter regimes
and data dimensions. Used to inform the choice of approximation in the main
likelihood computation.

## Requirements

```bash
pip install scipy matplotlib numpy
