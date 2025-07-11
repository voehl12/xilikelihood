import numpy as np 
import sys
import configparser
from calc_pdf import get_cf_nD
from random import randint
import time

time.sleep(randint(1,10))



j = int(sys.argv[1]) - 1

config = configparser.ConfigParser()
config.read('/cluster/home/veoehl/2ptlikelihood/config_1024.ini')
paths = config['Paths']
batchsize = int(config['Run']['batchsize'])
covs = np.load(paths['cov'])
cov = covs['matrix']
marrs = np.load(paths['M'])
mset = marrs['matrix']
allt = np.load(paths['t_sets'])['t']
batch_t = allt[j*batchsize:(j+1)*batchsize]
batch_cf = []

for i in range(len(batch_t)): 
    ts,cf = get_cf_nD(batch_t[i],mset,cov) 
    batch_cf.append(cf)
batch_cf = np.array(batch_cf)



np.savez(paths['result'] + 'job{:d}.npz'.format(j),ts=batch_t,cf=batch_cf)
