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
    from scipy.special import j0
    import scipy.integrate as integrate
    import grf_classes
    import matplotlib.pyplot as plt
    new_cl = grf_classes.TheoryCl(30,path='Cl_3x2pt_kids55.txt')
  
    
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
    assert np.allclose(nomask_bruteforce_cov-nomask_cov,np.zeros_like(nomask_cov)), (nomask_bruteforce_cov-nomask_cov)[np.argwhere(np.invert((np.isclose(nomask_bruteforce_cov-nomask_cov,np.zeros_like(nomask_cov),atol=1e-10))))]

    noise_cov = setup_cov.cov_xi(noise_sigma='default',pos_m=True,lmax=10)
    plt.figure()
    plt.imshow(noise_cov)
    plt.show()



    # next: constant C_l, pure noise implementation


def noise_test():
    import simulate
    nside = 256
    bins = [(n/10,(n+1)/10) for n in range(1,100)]
    simulate.xi_sim(0,None,bins,mask=None,mode='both',batchsize=1,sigma_n='default',nside=nside,
                    testing=True)

def noise_corrs():
    import simulate
    import grf_classes
    import matplotlib.pyplot as plt
    nside = 256
    kids55_cl = grf_classes.TheoryCl(3*nside-1,path='Cl_3x2pt_kids55.txt')
    
    bins = [(n/10,(n+1)/10) for n in range(1,100)]
    maskareas = [1000,4000,10000,'fullsky']
    scalecuts = [30,100,3*nside-1]
    linestyles = ['dotted','dashed','solid','dashdot']
    i = 0
    fig,( ax1,ax2 )= plt.subplots(1,2,figsize=(18,8))
    for m in maskareas:
        circ_mask = grf_classes.SphereMask([2],circmaskattr=(m,nside))
        color='C{:d}'.format(i)
        i += 1
        for ls,lm in enumerate(scalecuts):
            
            
            simulate.xi_sim(0,None,bins,mask=circ_mask.mask,mode='both',batchsize=1,sigma_n='default',nside=nside,noiselmax=lm,testing=False)
            sims = np.load('simulations/job0.npz')
            corr_namaster = sims['xip_n'][0,0]
            corr_treecorr = sims['xip_t'][0]
            bins2 = sims['theta']
            angsep = [(bins2[k,0] + bins2[k,1])/2 for k in range(len(bins2))]
            if i == 1:
                ax1.plot(angsep,corr_namaster,color=color,linestyle=linestyles[ls],label=lm)
                ax1.plot(angsep,corr_treecorr,color=color,alpha=0.5,linestyle=linestyles[ls])
                ax1.set_xlim(1,10)
            if ls == 0:
                ax2.plot(angsep,corr_namaster,color=color,linestyle=linestyles[ls],label='{} sqd'.format(str(m)))
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



def test_xip_pdf():
    import calc_pdf, grf_classes
    import matplotlib.pyplot as plt
    angbin = 2.5
    lmax = 30
    mask = grf_classes.SphereMask([2],circmaskattr=(1000,256),lmax=lmax)
    kids55_cl = grf_classes.TheoryCl(lmax,path='Cl_3x2pt_kids55.txt')
    x,pdf,norm = calc_pdf.pdf_xi_1D(angbin,c_ell_object=kids55_cl,lmax=lmax,kind='p',mask=mask,steps=2048,savestuff=True)
    #xnoisel,pdfnoisel,norm = calc_pdf.pdf_xi_1D(angbin,c_ell_object=kids55_cl,sigma_n=1.0,lmax=lmax,kind='p',mask=mask,steps=2048,savestuff=True)
    #xnoises,pdfnoises,norm = calc_pdf.pdf_xi_1D(angbin,c_ell_object=kids55_cl,sigma_n=0.23,lmax=lmax,kind='p',mask=mask,steps=2048,savestuff=True)
    xnoised,pdfnoised,norm = calc_pdf.pdf_xi_1D(angbin,c_ell_object=kids55_cl,sigma_n='default',lmax=lmax,kind='p',mask=mask,steps=2048,savestuff=True)
    return x,xnoised,pdf,pdfnoised
    """ plt.figure()
    plt.plot(x,pdf,label=r'no noise')
    plt.plot(xnoisel,pdfnoisel,label=r'$\sigma_{\epsilon} = 1.0$')
    plt.plot(xnoised,pdfnoised,label=r'$\sigma_{\epsilon} = 0.4$')
    plt.plot(xnoises,pdfnoises,label=r'$\sigma_{\epsilon} = 0.23$')
    plt.xlim(-1e-6,3e-6)
    plt.legend()
    plt.savefig('lh_xip_{:d}_{:d}deg_lm30_fullsky.png'.format(*angbin))
    plt.show()
 """

def test_gaussian_cov(theta,binsize):
    import grf_classes,setup_cov
    import scipy.stats as stats
    import matplotlib.pyplot as plt 
    nside = 256
    lmax = 3*nside - 1
    kids55_cl = grf_classes.TheoryCl(lmax,path='Cl_3x2pt_kids55.txt')
    
    maskarea = 1000
    fsky = maskarea/41253
    mean, cov = setup_cov.cov_xi_gaussian(kids55_cl,fsky,theta,theta)
    meann, covn = setup_cov.cov_xi_gaussian(kids55_cl,fsky,theta,theta,sigma_e='default',nside=nside,binsize_in_arcmin=binsize)
    
    x = np.linspace(-2e-6,6e-5,1000)
    return x, stats.norm.pdf(x,mean,np.sqrt(cov)), stats.norm.pdf(x,meann,np.sqrt(covn))

def noise_level_scale():   
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    angs = np.logspace(np.log10(0.5),np.log10(300),9)
    print(angs)
    for i,ang in enumerate(angs[3:]):
        binsize_in_arcmin = ang
        color = 'C{:d}'.format(i)
        x, pdf,pdfnoise = test_gaussian_cov(ang/60,binsize_in_arcmin)
        
    

        plt.plot(x,pdf,label=r'$\theta = {:3.1f}\degree$'.format(ang/60),color=color)
        plt.plot(x,pdfnoise/np.max(pdfnoise)*np.max(pdf),color=color,linestyle='dashed')
        
    plt.legend()
    plt.savefig('gaussiancov_noise_lmax.png')
    plt.show()


def gauss_exact_comp():
    import matplotlib.pyplot as plt
    x,xnoise,pdf,pdfnoise = test_xip_pdf()
    gaussx, gausspdf, gausspdfn = test_gaussian_cov(2.5)
    print(np.trapz(gausspdf,x=gaussx))
    plt.figure()
    plt.plot(x,pdf,label=r'no noise',color='C0')
    plt.plot(gaussx,gausspdf,label=r'no noise, Gaussian',color='C0',linestyle='dashed')
    plt.plot(xnoise,pdfnoise,label=r'$\sigma_{\epsilon} = 0.4$',color='C1')
    plt.plot(gaussx,gausspdfn,label=r'$\sigma_{\epsilon} = 0.4$, Gaussian',color='C1',linestyle='dashed')
    plt.xlim(-1e-6,5e-6)
    plt.legend()
    #plt.savefig('gauss_comparison_circ1000_lm30.png')
    plt.show()


noise_level_scale()





