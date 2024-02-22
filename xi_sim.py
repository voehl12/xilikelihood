from simulate import TwoPointSimulation
import sys

jobnumber = int(sys.argv[1])

new_sim = TwoPointSimulation([(1,2),(4,6)], maskname='kids_lowres',maskpath='/cluster/home/veoehl/2ptlikelihood/singlet_lowres.fits',l_smooth_mask=30,clname='3x2pt_kids_55', clpath="Cl_3x2pt_kids55.txt", batchsize=100,simpath="/cluster/scratch/veoehl/xi_sims",sigma_e='default')
new_sim.xi_sim_1D(jobnumber,save_pcl=True)

#'maskpath' : 'singlet_lowres.fits', 'maskname' : 'kids_lowres',
