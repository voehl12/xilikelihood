# tests for the field class:
import numpy as np
# should calculate Wllmm. 
#For no mask, these should be trivial (no mixing). 
#Could also add test case for circular mask with 1000 sqd. 
# Size should correspond to given ell_max
def test_palm_matching():
    import cov_calc
    palm_kinds = ['ReE', 'ImE', 'ReB', 'ImB']
    print(cov_calc.match_alm_inds(palm_kinds))
    assert cov_calc.match_alm_inds(palm_kinds) == [0,1,2,3], 'Function no work'


def test_cl_class():
    import field_class
    new_cl = field_class.TheoryCl(30)
    print(new_cl.ne,len(new_cl.ne))
    assert np.allclose(new_cl.ee,np.zeros(31)), 'Something wrong with zero Cl assignment'

def test_cov_xi():
    import setup_cov,field_class
    covs = np.load('precalc/cov_xip_l30_n256_circ1000.npz')
    kids55_cl = field_class.TheoryCl(30,path='Cl_3x2pt_kids55.txt')
    circ_mask = field_class.SphereMask([2],circmaskattr=(1000,256),lmax=30) # should complain if wpm_arr is None
    
    test_cov = setup_cov.cov_xi(kids55_cl,mask_object=circ_mask,pos_m=True)
    assert np.allclose(covs,test_cov), 'covariance calculation wrong'

test_cov_xi()