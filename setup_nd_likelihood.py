# setup n dimensional exact likelihood: combination matrices, covariance matrix and t/x grids
# write config file with all info, saves covariance and combination matrices
# as well as xi range

import numpy as np
import helper_funcs
import calc_pdf
import setup_m
from cov_setup import Cov
import configparser
import file_handling


def setup_config():
    config = configparser.ConfigParser()
    config.optionxform = str
    return config


def save_config(config):
    steps = config["Run"]["steps"]
    configname = config["Paths"]["result"] + "config_{:d}.ini".format(int(steps))
    with open(configname, "w") as configfile:
        config.write(configfile)


def setup_filenames(config, cov_path, m_path, tset_path, save_path):
    config["Paths"] = {"cov": cov_path, "M": m_path, "t_sets": tset_path, "result": save_path}


def setup_cls(config, cl_paths, cl_names, noise_contribs):
    config["Theory"] = {"n_cl": len(cl_paths)}
    for i in range(len(noise_contribs)):
        config.set("Theory", "noise{:d}".format(i), str(noise_contribs[i]))
    for i in range(len(cl_names)):
        config.set("Theory", cl_names[i], cl_paths[i])


def setup_covariances(area, nside, l_exact, ell_buffer, noise_contribs, clnames, clpaths):
    covs = [
        Cov(
            l_exact,
            [2],
            circmaskattr=(area, nside),
            clpath=clpath,
            clname=clname,
            sigma_e=noise,
            l_smooth_mask=30,
            l_smooth_signal=None,
            cov_ell_buffer=ell_buffer,
        )
        for clpath, clname, noise in zip(clpaths, clnames, noise_contribs)
    ]
    return covs


def setup_likelihood(config, covs, combs, ang_bins_in_deg, steps=1024):
    # split in geometry and cosmology?

    ndim = len(combs)
    config["Run"] = {"steps": str(steps), "ndim": str(ndim)}

    if not helper_funcs.check_property_equal(covs, "area"):
        raise RuntimeError("Areas are not equal.")
    if not helper_funcs.check_property_equal(covs, "nside"):
        raise RuntimeError("nsides are not equal.")

    if not helper_funcs.check_property_equal(covs, "exact_lmax"):
        raise RuntimeError("exact lmax are not equal (problem for compatibility of matrices).")
    # can probably be set up for different ell max as well and might work but needs to be tested
    if not helper_funcs.check_property_equal(covs, "cov_ell_buffer"):
        raise RuntimeError("ell buffer is not equal.")

    area = covs[0].area
    nside = covs[0].nside
    config["Geometry"] = {"area": area, "nside": nside}

    for i in range(ndim):
        config.set("Geometry", "lower{:d}".format(i), str(ang_bins_in_deg[i][0]))
        config.set("Geometry", "upper{:d}".format(i), str(ang_bins_in_deg[i][1]))

    cov_mat = calc_pdf.cov_xi_nD(covs)

    l_exact = covs[0].exact_lmax
    ell_buffer = covs[0].cov_ell_buffer

    prefactors = helper_funcs.prep_prefactors(
        ang_bins_in_deg, covs[0].wl, covs[0].lmax, covs[0].lmax
    )

    m = setup_m.m_xi_cross(
        (prefactors[0, :, : l_exact + 1], prefactors[1, :, : l_exact + 1]), combs=combs
    )
    cov_path = config["Paths"]["cov"]
    m_path = config["Paths"]["M"]

    file_handling.save_matrix(cov_mat, cov_path, kind="cov")
    file_handling.save_matrix(m, m_path)

    cov_triang = calc_pdf.get_cov_triang(covs)
    xi_max = []
    for _, comb in enumerate(combs):
        inds = calc_pdf.get_cov_pos(comb)
        cov = cov_triang[inds[0]][inds[1]]
        cov.cl2pseudocl()
        xip_estimate, _ = helper_funcs.pcl2xi((cov.p_ee, cov.p_bb, cov.p_eb), prefactors, l_exact)
        xi_max.append(np.fabs(xip_estimate[0]) * 6)
    tset_path = config["Paths"]["t_sets"]
    _, t_sets, _, _ = calc_pdf.setup_t(xi_max, steps)
    np.savez(tset_path, t=t_sets)
    # ximax should go up to number of dimensions and save all
    combs_n = [calc_pdf.get_cov_n(comb) for comb in combs]
    config["Params"] = {
        "l_exact": l_exact,
        "l_buffer": ell_buffer,
    }
    for i in range(ndim):
        config.set("Params", "comb{:d}".format(i), str(combs_n[i]))
        config.set("Params", "ximax{:d}".format(i), str(xi_max[i]))


def setup_cluster_computation(config, jobnum, steps):
    ndim = int(config["Run"]["ndim"])
    batchsize = int(steps**ndim / jobnum)
    config["Run"] = {"jobnum": str(jobnum), "batchsize": str(batchsize)}
