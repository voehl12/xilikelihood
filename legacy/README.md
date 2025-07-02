# Legacy Code

This folder contains deprecated code preserved for scientific reproducibility.

## Contents

- `calc_pdf_v1.py`: Original PDF calculation functions used in first paper
- `first_paper_scripts/`: Analysis scripts for first paper results

## Usage

⚠️ **Warning**: This code is preserved for reproducibility only. 
For new analyses, use the main package modules.

## Paper References

- First paper: [\[Citation/DOI here\]](https://arxiv.org/abs/2407.08718.)
- Used commit: `git tag v1.0-first-paper`


# Legacy Analysis Code

## Failed Approaches (Learning Experiences)

### `setup_nd_likelihood.py` - N-dimensional grid approach
- **Goal**: Compute full n-dimensional characteristic function grids  
- **Problem**: Computational complexity scales as O(steps^n)
- **Lesson**: Led to development of copula-based approach
- **Status**: ❌ Abandoned - computationally infeasible

### `calc_pdf_v1.py`, `setup_m.py` - First paper methodology  
- **Goal**: 1D and 2D likelihood calculations
- **Status**: ✅ Published in Paper 1 (should be moved to papers/first_paper/)

## Migration Notes
- ⚠️ Most of this "legacy" code is actually Paper 1 methodology
- 🎯 True legacy: `setup_nd_likelihood.py` and config-based experiments
- 📝 TODO: Move Paper 1 methods to `papers/first_paper_method/`
