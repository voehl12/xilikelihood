# tests/conftest.py
import pytest
import numpy as np
from theory_cl import TheoryCl
from mask_props import SphereMask

@pytest.fixture
def sample_theory_cl():
    """Provide a sample TheoryCl object for testing."""
    return TheoryCl(30, path="Cl_3x2pt_kids55.txt")

@pytest.fixture
def sample_mask():
    """Provide a sample mask for testing."""
    return SphereMask([2], circmaskattr=(1000, 256))

@pytest.fixture
def sample_angular_bins():
    """Provide sample angular bins for testing."""
    return [(4, 6), (7, 10)]