from simulate import TwoPointSimulation
import sys

jobnumber = int(sys.argv[1])

new_sim = TwoPointSimulation([(4, 6),(7,10)], circmaskattr=(4000, 256),l_smooth=20, clpath="Cl_3x2pt_kids55.txt", batchsize=1000,simpath="/cluster/scratch/veoehl/xi_sims" )
new_sim.xi_sim(jobnumber)
