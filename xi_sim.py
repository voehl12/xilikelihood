from simulate import TwoPointSimulation
import sys

jobnumber = int(sys.argv[1])

new_sim = TwoPointSimulation([(4, 6),(7,10)], circmaskattr=(4000, 256),l_smooth_mask=30,l_smooth_signal=100, clpath="Cl_3x2pt_kids55.txt", batchsize=1000,simpath="/cluster/scratch/veoehl/xi_sims" )
new_sim.xi_sim(jobnumber)
