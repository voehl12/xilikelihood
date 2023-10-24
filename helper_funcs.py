import numpy as np
import healpy as hp



def get_pp_sigma_n(nside=256,sigma_e='default'):
    # sigma = sigma_e (mean intrinsic ellipticity) / sqrt(2*N_gal*A_pixel)
    # e.g. https://arxiv.org/abs/2306.04689, eq. 14
    # N_gal in 1 / arcmin**2, therefore pixelarea in arcmin**2
    #0.23 /np.sqrt(5 * hp.nside2pixarea(nside,degrees=True)*3600
    if sigma_e == 'default':
        sigma_e = 0.4
    return sigma_e /np.sqrt(2* 5 * hp.nside2pixarea(nside,degrees=True)*3600)

def get_global_sigma_n(nside,survey_area_in_sqd,ang_sep_in_deg,bin_size_in_deg,sigma_e='default'):
    pp_sigma = get_pp_sigma_n(nside,sigma_e)
    A_annulus = 2 * np.pi * ang_sep_in_deg * bin_size_in_deg
    npix = hp.nside2npix(nside)
    N_ppairs = npix**2 * survey_area_in_sqd * A_annulus / 41253**2

    return pp_sigma / np.sqrt(N_ppairs)