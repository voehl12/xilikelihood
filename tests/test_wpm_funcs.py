import numpy as np
import pytest

from xilikelihood import wpm_funcs


def test_smooth_cl_snapshot(snapshot):
    l = np.arange(6)
    smoothed = wpm_funcs.smooth_cl(l, l_smooth=30)
    snapshot.check(smoothed, rtol=0.0, atol=0.0)


def test_wigners_on_array_behavior():
    if not wpm_funcs.HAS_WIGNER:
        with pytest.raises(ImportError):
            wpm_funcs.wigners_on_array(2, 2, 0, 0, 5)
    else:
        w = wpm_funcs.wigners_on_array(2, 2, 0, 0, 5)
        assert len(w) == 6
