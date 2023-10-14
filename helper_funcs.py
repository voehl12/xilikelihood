import numpy as np
import healpy as hp



def get_sigma_n(nside=256):
    return 0.23 /np.sqrt(5 * hp.nside2pixarea(nside,degrees=True)*3600)