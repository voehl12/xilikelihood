import numpy as np
import plotting
import calc_pdf
import configparser
import file_handling




def load_config(configpath):
    config = configparser.ConfigParser()
    config.read(configpath)
    return config




def convert_nd_cf_to_pdf(config):
    t0_2,dt_2,_,_,cf_grid = file_handling.read_2D_cf(config)
    x_grid, pdf_grid = calc_pdf.cf_to_pdf_nd(cf_grid, t0_2, dt_2, verbose=True)
    return x_grid, pdf_grid


def compare_to_sims():
    allxi1,allxi2 = plotting.read_sims_nd(filepath, [1,2],  (4, 6),1000,exact_lmax)
    cov_estimate_low = np.cov(np.array([allxi1,allxi2]),ddof=1)
    pass


def compare_to_gaussian(cov_objects):
    mu,cov = calc_pdf.cov_xi_gaussian_nD(cov_objects,xi_combs=xi_combs,angbins_in_deg=angbins_in_deg, lmin=0,lmax=exact_lmax)
    pass

def get_marginal_likelihoods(x_grid,pdf_grid):
    pdf_53 = np.trapz(pdf_grid,x=x_grid[:,0,0],axis=0)
    pdf_55 =  np.trapz(pdf_grid,x=x_grid[0,:,1],axis=1)
    x_53, x_55 = x_grid[0,:,1], x_grid[:,0,0]
    mu_53 = np.trapz(x_53 * pdf_53,x=x_53)
    mu_55 = np.trapz(x_55 * pdf_55,x=x_55)
    pass

def normalize_pdfs():
    pass