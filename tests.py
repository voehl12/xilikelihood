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
    import grf_classes
    new_cl = grf_classes.TheoryCl(30)
    print(new_cl.ne,len(new_cl.ne))
    assert np.allclose(new_cl.ee,np.zeros(31)), 'Something wrong with zero Cl assignment'

def test_cov_xi():
    import setup_cov,grf_classes
    import matplotlib.pyplot as plt
    covs = np.load('../corrfunc_distr/cov_xip_l10_n256_circ1000.npz')
    cov_xip = covs['cov']
    kids55_cl = grf_classes.TheoryCl(10,path='Cl_3x2pt_kids55.txt')
    circ_mask = grf_classes.SphereMask([2],circmaskattr=(1000,256),lmax=10) # should complain if wpm_arr is None
    
    test_cov = setup_cov.cov_xi(kids55_cl,mask_object=circ_mask,pos_m=True)
    assert np.array_equal(cov_xip,test_cov), 'covariance calculation wrong'
    
    nomask_cov = setup_cov.cov_xi(kids55_cl,pos_m=True)
    diag = np.diag(nomask_cov)
    diag_arr = np.diag(diag)
    assert np.array_equal(nomask_cov,diag_arr)
    
    nomask_mask = grf_classes.SphereMask([2],circmaskattr=('fullsky',256),lmax=10)
    nomask_bruteforce_cov = setup_cov.cov_xi(kids55_cl,mask_object=nomask_mask,pos_m=True)
    plt.figure()
    plt.imshow(2*nomask_bruteforce_cov-nomask_cov)
    plt.colorbar()
    plt.figure()
    plt.plot(kids55_cl.ee)
    plt.show() 
    
    assert np.allclose(nomask_bruteforce_cov-nomask_cov,np.zeros_like(nomask_cov),atol=1e-10), (nomask_bruteforce_cov-nomask_cov)[np.argwhere(np.invert((np.isclose(nomask_bruteforce_cov-nomask_cov,np.zeros_like(nomask_cov),atol=1e-10))))]

    # next: constant C_l, pure noise implementation


def noise_test():
    import simulate
    nside = 256
    bins = [(n/10,(n+1)/10) for n in range(1,100)]
    simulate.xi_sim(0,None,bins,mask=None,mode='both',batchsize=1,sigma_n='default',nside=nside,testing=True)

def noise_corrs():
    import simulate
    import grf_classes
    import matplotlib.pyplot as plt
    nside = 512
    kids55_cl = grf_classes.TheoryCl(3*nside-1,path='Cl_3x2pt_kids55.txt')
    
    bins = [(n/10,(n+1)/10) for n in range(1,100)]
    maskareas = [1000,4000,10000]
    scalecuts = [30,100,3*nside-1]
    linestyles = ['dotted','dashed','solid']
    i = 0
    fig,( ax1,ax2 )= plt.subplots(1,2,figsize=(18,8))
    for m in maskareas:
        circ_mask = grf_classes.SphereMask([2],circmaskattr=(m,nside))
        color='C{:d}'.format(i)
        i += 1
        for ls,lm in enumerate(scalecuts):
            
            
            simulate.xi_sim(0,None,bins,mask=circ_mask.mask,mode='both',batchsize=1,sigma_n='default',nside=nside,noiselmax=lm,testing=True)
            sims = np.load('simulations/job0.npz')
            corr_namaster = sims['xip_n'][0,0]
            corr_treecorr = sims['xip_t'][0]
            bins2 = sims['theta']
            angsep = [(sims['theta'][k,0] + sims['theta'][k,1])/2 for k in range(len(bins2))]
            if i == 1:
                ax1.plot(angsep,corr_namaster,color=color,linestyle=linestyles[ls],label=lm)
                ax1.plot(angsep,corr_treecorr,color=color,alpha=0.5,linestyle=linestyles[ls])
                ax1.set_xlim(1,10)
            if ls == 2:
                ax2.plot(angsep,corr_namaster,color=color,linestyle=linestyles[ls],label='{:d} sqd'.format(m))
                ax2.plot(angsep,corr_treecorr,color=color,alpha=0.5,linestyle=linestyles[ls])
    ax1.set_xlim(1,10)
    ax1.set_ylim(-2e-6,2e-6)
    ax1.axhline(y=1e-7, color="black", linestyle="--")
    ax2.axhline(y=1e-7, color="black", linestyle="--")
    ax2.set_xlim(1,10)
    ax2.set_ylim(-2e-6,2e-6)
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel(r'$\theta$ [deg]')
    ax2.set_xlabel(r'$\theta$ [deg]')
    ax1.set_ylabel(r'$\xi^+$')
    ax2.set_ylabel(r'$\xi^+$')
    plt.savefig('noise_correlation_nside{:d}.pdf'.format(nside))
    plt.show()




    
  
    
    

    
    
   
    # mask, fullsky, pure noise
    # on the full sky, treecorr underestimates, or namaster too high? check normalization factor
    #assert np.allclose(sims['xip_n'][:,0,0],sims['xip_t'][:,0],atol=1e-07), 'treecorr and namaster do not agree'


test_cov_xi()
