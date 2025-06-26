import os
import numpy as np
import helper_funcs
import glob
import re  # Import regular expressions


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

def xi_sims_from_pcl(i,prefactors,filepath,lmax=None):
    
    pclfile = np.load(filepath+"/pcljob{:d}.npz".format(i))
    assert lmax <= int(pclfile["lmax"]), f"requested lmax too high for simulated pcljob {i}. Need new simulations."
    pcl_s = np.array([pclfile['pcl_e'],pclfile['pcl_b'],pclfile['pcl_eb']])
    xips,xims = helper_funcs.pcls2xis(pcl_s,prefactors,out_lmax=lmax)
    

    return xips,xims

def read_sims_nd(filepath, njobs, lmax, kind="xip", prefactors=None, theta=None):
    # Make this truly nd
    all_xi = []
    missing = []
    angles = None  # To store angles from the first file

    for i in range(1, njobs + 1):
        if os.path.isfile(filepath + "/job{:d}.npz".format(i)):
            xifile = np.load(filepath + "/job{:d}.npz".format(i))
            try:
                #assert lmax == int(xifile["lmax"])
                xi = xifile[kind]  # Shape (batchsize,n_corr,n_theta)
                
            except AssertionError:
                print(f"lmax mismatch in job {i}. Generating xis using pcls2xis.")
                pclfile = np.load(filepath + "/pcljob{:d}.npz".format(i))
                
                
                # Check for prefactors in xifile.files
                if "prefactors" in xifile.files:
                    prefactors = xifile["prefactors"]
                elif prefactors is None:
                    raise ValueError("Prefactors must be provided for older simulations.")
                assert len(prefactors) == len(theta), "Prefactors and theta must have the same length."
                xips, xims = xi_sims_from_pcl(i, prefactors, filepath, lmax=lmax)
                
                xi = xips if kind == "xip" else xims

                # Update the filepath for storing the new xis
                new_folder = filepath.replace(
                    "llim_{}".format(xifile["lmax"]),
                    "llim_{}".format(lmax),
                )
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                new_filepath = new_folder + "/job{:d}.npz".format(i)
                np.savez(
                    new_filepath,
                    xip=xips,
                    xim=xims,
                    lmax=lmax,  # Update lmax in the saved file
                    theta=theta,
                )
            if angles is None:
                angles = [tuple(angle) for angle in xifile["theta"]]  # Convert to list of 2-tuples
                print(angles)
            # Assert that theta (if provided) is part of angles
            if theta is not None:
                
                assert set(theta).issubset(set(angles)), "Provided theta contains angles not in the dataset."
            else:
                theta = angles  # Use all angles if none provided
            all_xi.append(xi)
        else:
            print("Missing job number {:d}.".format(i))
            missing.append(i)

    all_xi = np.concatenate(all_xi, axis=0)

    missing_string = ",".join([str(x) for x in missing])
    print(missing_string)
    return all_xi, theta  # Return all xi and the saved angles

def read_pcl_sims(filepath,njobs):
    allpcl=[]
    missing = []
    for i in range(1,njobs+1):
        print(i)
        if os.path.isfile(filepath+"/pcljob{:d}.npz".format(i)):
            pclfile = np.load(filepath+"/pcljob{:d}.npz".format(i))
            pcl_s = np.array([pclfile['pcl_e'],pclfile['pcl_b'],pclfile['pcl_eb']])
            pcl_s = np.swapaxes(pcl_s,0,1)
            allpcl += list(pcl_s)
        else:
            print('Missing job number {:d}.'.format(i))
            missing.append(i)
    allpcl = np.array(allpcl)
    print(allpcl.shape)
    #allpcl.reshape(allpcl.shape[0]*allpcl.shape[1],3,768)
    return allpcl

def read_posterior_files(pattern, regex=None):
    """
    Reads files matching a given pattern and appends their values to lists.

    Parameters:
    - pattern: A string pattern to match files (e.g., 's8posts/s8post_*.npz').
    - regex: A regular expression to further filter matched files (optional).

    Returns:
    - A dictionary containing lists of values extracted from the files.
    """
    gauss_posteriors, exact_posteriors, s8, means, combs, available = [], [], [], [], [], []
    
    # Use glob to find files matching the pattern
    files = glob.glob(pattern)
    
    # If a regex is provided, filter files using it
    if regex:
        files = [f for f in files if re.search(regex, f)]
    
    for file in files:
        try:
            posts = np.load(file)
            gauss_post, exact_post = posts['gauss'], posts['exact']
            gauss_posteriors.append(gauss_post.flatten())
            exact_posteriors.append(exact_post.flatten())
            s8.append(posts['s8'].flatten())
            #means.append(posts['means'])
            #combs.append(posts['comb'])
            available.append(True)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            available.append(False)
    
    return {
        "gauss_posteriors": np.array(gauss_posteriors).flatten(),
        "exact_posteriors": np.array(exact_posteriors).flatten(),
        "s8": np.array(s8).flatten(),
        #"means": np.array(means),
        #"combs": np.array(combs),
        "available": available
    }





def xi_sims_from_pcl(i,prefactors,filepath,lmax=None):
    
    pclfile = np.load(filepath+"/pcljob{:d}.npz".format(i))
    pcl_s = np.array([pclfile['pcl_e'],pclfile['pcl_b'],pclfile['pcl_eb']])
    xips,xims = pcls2xis(pcl_s,prefactors,out_lmax=lmax)
    
# make dictionary of initial xi-file and add this angbin and these xis
    return xips,xims




def read_xi_sims(filepath,njobs,angbins,kind="xip",prefactors=None,lmax=None):
    raise DeprecationWarning("This function is deprecated. Use read_sims_nd instead.")
     
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



