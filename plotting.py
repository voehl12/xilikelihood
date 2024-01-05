import matplotlib.pyplot as plt
import numpy as np
import os


def read_sims(filepath,njobs,angbin,kind="xip"):
    allxi=[]
    for i in range(1,njobs+1):
        if os.path.isfile(filepath+"/job{:d}.npz".format(i)):
            xifile = np.load(filepath+"/job{:d}.npz".format(i))
            angs = xifile["theta"]
            angind = np.where(angs == angbin)
            xip = xifile[kind][:,angind[0][0]]
            allxi.append(xip)
        else:
            print('Missing job number {:d}.'.format(i))
    allxi = np.array(allxi)
    allxi = allxi.flatten()
    return allxi



def plot_hist(ax,sims, name, color='C0', linecolor='C3', exact_pdf=None, label=False, fit_gaussian=False):

    
    #


    if label:
        n, bins, patches = ax.hist(
            sims, bins=500, density=True, facecolor=color, alpha=0.5, label=name, color=color
        )
    else:
        n, bins, patches = ax.hist(sims, bins=500, density=True, facecolor=color, alpha=0.5)

    if exact_pdf is not None:
        x, pdf = exact_pdf
        ax.plot(x, pdf, color=linecolor, linewidth=2)
    ax.set_xlabel((r"$\xi^+$"))
    ax.legend()
    ax.ticklabel_format(style="scientific", scilimits=(0, 0))
    return ax



