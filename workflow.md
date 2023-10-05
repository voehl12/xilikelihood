
SIMS = True
PDF = True

lmax

load theory C_ell -> TheoryCl class
load mask. single function file2mask. only call once in the beginning. everything else takes mask array or mask object. mask class has the read function - maybe just create an object and pass MaskClass.mask to simulations. 
this also gives nside directly.  
create field_class

corr_func
    binning or angles
    set up m matrix

pseudo_Cell
    binning?
    set up m matrix for given covariance matrix structure (different than uphams)
    l_out for covariance matrix. For higher ell should not calculate covariance for all other ell as well. 
    need new covariance matrix for crosscorrelation of different ells

noise 

create setup file that can be passed to sims or pdf calc? cell and mask file should be shared, angles and bins or multipole moments as well. noise as well. 

if SIMS:
    nsims -> batchsize. could create a euler or laptop method for this. 
    treecorr or namaster
    run sims

if PDF: 
    calc pdf
        calc wllmm
        calc m: setup_m.py; need to add read and write functionality
        calc covariance matrix: setup_cov.py; imports cov_calc.py, need to finish read and write functionality


