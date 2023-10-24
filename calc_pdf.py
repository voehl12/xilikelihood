import setup_cov,setup_m, cf_to_pdf
import numpy as np

def pdf_xi_1D(angle_in_deg,c_ell_object=None,sigma_n=None,lmax=None,kind='p',mask=None,steps=2048,savestuff=True,recalc_cov=False):
    if c_ell_object is None:
        clname = 'none'

        if lmax is None:
            raise RuntimeError('pdf_xi_1D: lmax needs to be given for pure noise maps')
    else:
        lmax = c_ell_object.lmax
        clname = c_ell_object.name
    if mask is None:
        nside = 1
        maskname = 'fullsky'
    else:
        nside = mask.nside
        maskname = mask.name
    if type(angle_in_deg) is tuple:
        ang = '{:.0f}_{:.0f}'.format(angle_in_deg[0],angle_in_deg[1])
    else:
        ang = '{:.0f}'.format(angle_in_deg)

 
    

    if sigma_n is not None:
        if isinstance(sigma_n,str):
            sigmae = sigma_n
        else:
            sigmae = str(sigma_n)
            sigmae = sigmae.replace('.','')
        covname = 'cov_xi{}_l{:d}_n{:d}_{}_{}_noise{}.npz'.format(kind,lmax,nside,maskname,clname,sigmae)
        mname = 'm_xi{}_l{:d}_t{}_{}.npz'.format(kind,lmax,ang,maskname)
    else:
        covname = 'cov_xi{}_l{:d}_n{:d}_{}_{}.npz'.format(kind,lmax,nside,maskname,clname)
        mname = 'm_xi{}_l{:d}_t{}_{}.npz'.format(kind,lmax,ang,maskname)
    



    if not setup_cov.check_cov(covname) or recalc_cov == True:
        cov = setup_cov.cov_xi(c_ell_object,mask_object=mask,pos_m=True,sigma_e=sigma_n,lmax=lmax)
        if savestuff:
            setup_cov.save_cov(cov,covname)

    else: 
        cov = setup_cov.load_cov(covname)
   
        

    if setup_m.check_m(mname):
        m = setup_m.load_m(mname)
    else:
        m = setup_m.mmatrix_xi(angle_in_deg,mask_object=mask,kind=kind)
        if savestuff:
            setup_m.save_m(m,mname)

    prod = cov @ m
    evals = np.linalg.eigvals(prod)
    
    xip_max = 6.0e-6
    dt_xip = 0.45 * 2 * np.pi / xip_max
    
    t0 = -0.5 * dt_xip * (steps - 1)
    t = np.linspace(t0, -t0, steps - 1)


    t_evals = np.tile(evals,(steps-1,1)) # number of t steps rows of number of eigenvalues columns; each column contains one eigenvalue
    tmat = np.repeat(t,len(evals)) # repeat each t number of eigenvalues times
    tmat = np.reshape(tmat,(steps-1,len(evals))) # recast to array with number of t steps rows and number of eigenvalues columns; each row contains one t step
    t_evals *= tmat # each row is one set of eigenvalues times a given t step -> need to multiply along this row (axis 1)


    cf = np.prod(np.sqrt(1/(1 - 2 * 1j * t_evals)),axis=1) # gives t steps different values for characteristic function



    x, pdf = cf_to_pdf.cf_to_pdf_1d(cf, t0, dt_xip)
    norm = np.trapz(pdf,x=x)
    
    return x,pdf,norm
    




def pdf_pcl():
    pass
