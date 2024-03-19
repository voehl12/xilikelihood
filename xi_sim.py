from simulate import TwoPointSimulation
import sys

jobnumber = int(sys.argv[1])

new_sim = TwoPointSimulation([(2,3)],maskpath='/cluster/home/veoehl/2ptlikelihood/singlet_lowres.fits',maskname='kids_lowres',l_smooth_mask=30,clname='3x2pt_kids_55', clpath="Cl_3x2pt_kids55.txt", batchsize=1000,simpath="/cluster/scratch/veoehl/xi_sims/",sigma_e='default',ximode='treecorr')
new_sim.xi_sim_1D(jobnumber,save_pcl=False,pixwin=False,plot=False)

#'maskpath' : 'singlet_lowres.fits', 'maskname' : 'kids_lowres',
# maskpath='/cluster/home/veoehl/2ptlikelihood/singlet_lowres.fits',maskname='kids_lowres',