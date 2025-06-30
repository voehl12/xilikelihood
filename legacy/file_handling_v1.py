"""
Legacy file I/O functions for old paper plots and analyses.

These functions are preserved for compatibility with existing paper plots
but should not be used for new analyses.
"""

import numpy as np
import os
import traceback
import warnings

def read_xi_sims(filepath, njobs, angbins, kind="xip", prefactors=None, lmax=None):
    """
    LEGACY: Read xi simulations for specific angular bins.
    
    This function is preserved for compatibility with old paper plots.
    For new analyses, use file_handling.read_sims_nd() instead.
    """
    warnings.warn(
        "read_xi_sims is legacy code. Use file_handling.read_sims_nd() for new analyses.",
        FutureWarning,
        stacklevel=2
    )


    allsims = []
    for j,angbin in enumerate(angbins):
        allxi=[]
        missing = []
        for i in range(1,njobs+1):
            
            if os.path.isfile(filepath+"/job{:d}.npz".format(i)):
                xifile = np.load(filepath+"/job{:d}.npz".format(i))
                angs = xifile["theta"]
                print(angs)
                if filepath[-2:] == 'rr':
                    angind = np.where(angs == np.mean(angbin))
                else:
                    angind = np.where(angs == angbin)
                print(angind)
                
                if lmax is not None or len(angind[0]) == 0:
                    try:
                        xip = xi_sims_from_pcl(i,prefactors,filepath,lmax=lmax)[0][:,j]
                    except FileNotFoundError:
                        print('Missing job number {:d}.'.format(i))
                        missing.append(i)
                        #traceback.print_exc()
                        xip =[]
                    except:
                        traceback.print_exc()
                        xip =[]
                else:
                    #xip = xifile[kind][:,angind[0][0]]
                    xip = xifile[kind][:,angind[0][0]]
                allxi += list(xip)
            else:
                print('Missing job number {:d}.'.format(i))
                missing.append(i)
        allxi = np.array(allxi)
        missing_string = ','.join([str(x) for x in missing])
        print(missing_string)
        allsims.append(allxi)
    #allxi = allxi.flatten()
    return np.array(allsims)


def read_2D_cf(config):
    ndim = 2  # int(config["Run"]['ndim'])
    paths = config["Paths"]
    params = config["Params"]
    batchsize = int(config["Run"]["batchsize"])
    numjobs = int(config["Run"]["jobnum"])
    steps = int(params["steps"])
    t_sets = np.load(paths["t_sets"])["t"]

    xi_max = [float(params["ximax{:d}".format(i)]) for i in range(1, ndim + 1)]

    t_inds, t_sets, t0_set, dt_set = helper_funcs.setup_t(xi_max, steps)

    cf_grid = np.full((steps - 1, steps - 1), np.nan, dtype=complex)

    resultpath = paths["result"]
    # resultpath = '/cluster/scratch/veoehl/2Dcf/xip_5535bins_new/'

    t0_2 = np.array(t0_set)
    dt_2 = np.array(dt_set)

    ind_sets = np.stack(np.meshgrid(t_inds, t_inds), -1).reshape(-1, 2)
    fail_list = []

    for i in range(numjobs):

        try:
            batch = np.load(resultpath + "job{:d}.npz".format(i))
            size = os.path.getsize(resultpath + "job{:d}.npz".format(i))
            # if size < 131560 and i != numjobs-1:
            #    print('removing file job{:d}.npz'.format(i))
            #    os.remove(resultpath + 'job{:d}.npz'.format(i))
            batch_t = batch["ts"]
            batch_cf = batch["cf"]
        except:
            fail_list.append(i)
            batch_t = t_sets[i * batchsize : (i + 1) * batchsize]
            batch_cf = np.zeros(len(batch_t))

        inds = ind_sets[i * batchsize : (i + 1) * batchsize]
        # high_ell_ext = calc_pdf.high_ell_gaussian_cf_nD(batch_t,mu,cov)

        # high_ell_ext[np.isnan(high_ell_ext)] = 0
        # high_ell_ext[np.isinf(high_ell_ext)] = 0

        for j, idx in enumerate(inds):
            try:
                cf_grid[tuple(idx)] = batch_cf[j]
            except:
                cf_grid[tuple(idx)] = 0

    missing_string = ",".join([str(x + 1) for x in fail_list])
    print(missing_string)

    print(len(fail_list))
    np.savez("missing_jobs.npz", numbers=np.array(fail_list))
    return t0_2, dt_2, t_sets, ind_sets, cf_grid


def load_pdfs(name):
    print("Loading pdfs ", end="")
    mfile = np.load(name)
    print("with angles ", end="")
    print(mfile["angs"])
    return mfile["x"], mfile["pdf"], mfile["stats"], mfile["angs"]


def load_cfs(name):
    print("Loading cfs ", end="")
    mfile = np.load(name)
    print("with angles ", end="")
    print(mfile["angs"])
    return mfile["t"], mfile["cf_re"], mfile["cf_im"], mfile["ximax"], mfile["angs"]