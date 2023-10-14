import setup_cov,setup_m

def pdf_xi_1D(angles_in_deg,c_ell_object=None,sigma_n=None,kind='p',mask=None,savestuff=True):
    if setup_cov.check_cov():
        cov = setup_cov.load_cov()
    else:
        cov = setup_cov.cov_xi(c_ell_object,mask_object=mask,pos_m=True,noise_sigma=sigma_n)
        if savestuff:
            setup_cov.save_cov()
    if setup_m.check_m():
        m = setup_m.load_m()
    else:
        m = setup_m.mmatrix_xi(angles_in_deg,mask_object,kind=kind)
        if savestuff:
            setup_m.save_m()

    prod = cov @ mmatrix
    evals = np.linalg.eigvals(prod)
    
    xip_max = 3.0e-8
    dt_xip = 0.45 * 2 * np.pi / xip_max
    steps = 2048
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
