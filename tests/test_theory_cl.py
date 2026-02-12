import numpy as np


def test_cl_class():
    from xilikelihood import theory_cl

    new_cl = theory_cl.TheoryCl(30)
    assert np.allclose(new_cl.ee, np.zeros(31))


def test_redshift_bin_z_eff(sample_redshift_bins):
    bin0 = sample_redshift_bins[0]
    expected = np.sum(bin0.z * bin0.nz) / np.sum(bin0.nz)
    assert np.isclose(bin0.z_eff, expected)


def test_bin_combination_mapper():
    from xilikelihood.theory_cl import BinCombinationMapper

    mapper = BinCombinationMapper(3)
    assert mapper.n_combinations == 6
    idx = mapper.get_index((2, 1))
    assert idx == mapper.get_index((1, 2))
    assert mapper.get_combination(idx) == (2, 1)


def test_prepare_theory_cl_inputs(sample_redshift_bins):
    from xilikelihood.theory_cl import prepare_theory_cl_inputs

    numerical_combinations, redshift_bin_combinations, is_cov_cross, shot_noise, mapper = (
        prepare_theory_cl_inputs(sample_redshift_bins, noise="default")
    )

    assert len(numerical_combinations) == 3
    assert len(redshift_bin_combinations) == 3
    assert list(is_cov_cross) == [False, False, True]
    assert shot_noise == ["default","default", None]
    assert mapper.n_combinations == 3


def test_generate_theory_cl_sigma_e(sample_redshift_bins):
    from xilikelihood.theory_cl import generate_theory_cl, prepare_theory_cl_inputs

    _, redshift_bin_combinations, is_cov_cross, shot_noise, _ = (
        prepare_theory_cl_inputs(sample_redshift_bins, noise="default")
    )

    theory_cls = generate_theory_cl(
        lmax=10,
        redshift_bin_combinations=redshift_bin_combinations,
        shot_noise=shot_noise,
        cosmo=None,
        nside=None,
    )

    assert len(theory_cls) == 3
    for cl, expected_sigma_e in zip(theory_cls, shot_noise):
        assert cl.sigma_e == expected_sigma_e