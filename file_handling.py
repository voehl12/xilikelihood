import os
import numpy as np
import helper_funcs


class File:
    """
    _summary_
    """

    def __init__(self, filename, kind="file") -> None:
        self.path = filename
        self.kind = kind
        pass

    def check_for_file(self):
        print("Checking for {}s...".format(self.kind))
        if os.path.isfile(self.path):
            print("Found some.")
            return True
        else:
            print("None found.")
            return False


def check_for_file(name, kind="file"):
    print("Checking for {}s...".format(kind))
    if os.path.isfile(name):
        print("Found some.")
        return True
    else:
        print("None found.")
        return False


# load can become one function with a dictionary or list of what to load and return
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


def save_matrix(m, filename, kind="M"):
    print("Saving {} matrix.".format(kind))
    np.savez(filename, matrix=m)


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


def read_sims_nd(filepath, corr_num, angbin, njobs, lmax, kind="xip"):
    # should make this truly nd
    allxi1, allxi2 = [], []
    missing = []
    for i in range(1, njobs + 1):
        if os.path.isfile(filepath + "/job{:d}.npz".format(i)):
            xifile = np.load(filepath + "/job{:d}.npz".format(i))
            assert lmax == int(xifile["lmax"])
            angs = xifile["theta"]
            angind = np.where(angs == angbin)
            xip1, xip2 = (
                xifile[kind][:, corr_num[0], angind[0][0]],
                xifile[kind][:, corr_num[1], angind[0][0]],
            )
            allxi1.append(xip1)
            allxi2.append(xip2)
        else:
            print("Missing job number {:d}.".format(i))
            missing.append(i)
    allxi1 = np.concatenate(allxi1, axis=0)
    allxi2 = np.concatenate(allxi2, axis=0)

    missing_string = ",".join([str(x) for x in missing])
    print(missing_string)
    print(len(missing))
    return np.array([allxi1, allxi2])
