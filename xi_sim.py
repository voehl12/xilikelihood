from simulate import TwoPointSimulation
import sys

jobnumber = int(sys.argv[1])

new_sim = TwoPointSimulation([(1,2),(4,6)], circmaskattr=(1000, 256),l_smooth_mask=30,clname='3x2pt_kids_55', clpath="Cl_3x2pt_kids55.txt", batchsize=1000,simpath="/cluster/scratch/veoehl/xi_sims",sigma_e='default')
new_sim.xi_sim_1D(jobnumber)

#'maskpath' : 'singlet_lowres.fits', 'maskname' : 'kids_lowres',
