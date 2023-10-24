import numpy as np 
import healpy as hp
import wpm_funcs 
import pickle


def save_maskobject(maskobject,dir=''):
    name = dir + maskobject.name + '_l' + str(maskobject.lmax) + '_n' + str(maskobject.nside)
    maskfile = open(name, 'wb') 
    pickle.dump(maskobject, maskfile)

class TheoryCl:
    def __init__(self,lmax,path=None,theory_lmin=2,clname='3x2pt_kids'):
        self.ell = np.arange(lmax+1)
        self.len_l = len(self.ell)
        self.theory_lmin = theory_lmin
        self.lmax = lmax
        self.nn = None
        self.ee = None
        self.ne = None
        self.bb = None
        self.path = None
        if path:
            self.path = path
            self.name = clname
            self.read_file()
            self.load_cl()
            print('Loaded C_l with lmax = {:d}'.format(self.lmax))
        else:
            print('Warning: no theory Cl provided, calculating with Cl=0')
            self.set_cl_zero()

        
        

    def read_file(self):
        self.raw_spectra = np.loadtxt(self.path)

    def load_cl(self):
        spectra = np.concatenate((np.zeros((3,self.theory_lmin)), self.raw_spectra[:,:self.lmax - self.theory_lmin + 1]),axis=1)
        self.ee = spectra[0]
        self.ne = spectra[1]
        self.nn = spectra[2]

    def set_cl_zero(self):
        self.name = 'none'
        self.ee = self.ne = self.nn = np.zeros(self.len_l)


  

class SphereMask:
    # possibility to use different masks for different fields? -> NO, that would defeat the purpose of a class for 
    # a given mask. Rather have several instances. 
    def __init__(self,spins,mask_path=None,circmaskattr=None,prep_wlm=True,lmin=None,lmax=None,maskname='mask') -> None:
        """
        circmaskattr: tuple (area,nside) for a circular mask
        
        """
        self.name = maskname
        if mask_path is not None:
            self.mask_path = mask_path
            self.read_file()
            
        elif circmaskattr is not None:
            if circmaskattr[0] == 'fullsky':
                self.nside = circmaskattr[1]
                self.fullsky_mask()
                
            else:

                self.area, self.nside = circmaskattr
                self.create_circmask()
        else:
            raise RuntimeError('Please specify either a mask path or attributes for a circular mask')

        self.spins = spins
        self.spin0 = None
        self.spin2 = None
        self.n_field = 0
        if 0 in spins:
            self.spin0 = True
            self.n_field += 1
        if 2 in spins:
            self.spin2 = True
            self.n_field += 2
        if not self.spin0 and not self.spin2:
            raise RuntimeError('Spin needs to be 0 and/or 2')
        
        
        if lmax:
            self.lmax = lmax
        else:
            self.lmax = 3*self.nside - 1
            print('Warning: lmax has been set to {:d}.'.format(self.lmax))
        if lmin:
            self.lmin = lmin
        else:
            self.lmin = 0
        if prep_wlm:
            self.wlm = self.mask2wlm()
        self.L = None
        self.M = None
        self.w0_arr = None
        self.wpm_arr = None
        self.w_arr = None

    def read_file(self):
        self.mask = hp.fitsfunc.read_map(self.mask_path, verbose=True)
        self.nside = hp.pixelfunc.get_nside(self.mask)     
        # self.area (calculate unmasked area from pixel sizes)

    def create_circmask(self):
        # implement loading of such a mask, rename function to get_circmask
        npix = hp.nside2npix(self.nside)
        m = np.zeros(npix)
        vec = hp.ang2vec(np.pi / 2, 0)
        r = np.sqrt(self.area / np.pi)
        disc = hp.query_disc(nside=self.nside, vec=vec, radius=np.radians(r))
        m[disc] = 1
        self.mask = m

        self.mask_path = 'circular_{:d}sqd_nside{:d}.fits'.format(self.area,self.nside)
        self.name = 'circ{:d}'.format(self.area)
        hp.fitsfunc.write_map(self.mask_path,m,overwrite=True)
        
    def fullsky_mask(self):
        npix = hp.nside2npix(self.nside)
        m = np.ones(npix)
        self.mask = m
        self.mask_path = 'fullsky_nside{:d}.fits'.format(self.nside)
        self.name = 'fullsky'
        hp.fitsfunc.write_map(self.mask_path,m,overwrite=True)

    def mask2wlm(self):
        """
        Calculate spherical harmonics of the mask. 
        """
        return hp.sphtfunc.map2alm(self.mask, lmax=self.lmax)      

    def initiate_w_arrs(self):
        self.L = np.arange(self.lmax+1)
        self.M = np.arange(-self.lmax,self.lmax+1)
        Nl = len(self.L)
        Nm = len(self.M)
        if self.spin0:
            self.w0_arr = np.zeros((Nl,Nm,Nl,Nm),dtype=complex)
        if self.spin2:
            self.wpm_arr = np.zeros((2,Nl,Nm,Nl,Nm),dtype=complex)

    def calc_w_element(self,L1,L2,M1,M2): # move this to wpm_funcs, since it does not attach anything to the object? but it heavily relies on wlm
        
        m = M1-M2
        m1_ind = np.argmin(np.fabs(M1-self.M))
        m2_ind = np.argmin(np.fabs(M2-self.M))
        l1_ind = np.argmin(np.fabs(L1-self.L))
        l2_ind = np.argmin(np.fabs(L2-self.L))
        inds = (l1_ind,m1_ind,l2_ind,m2_ind)
        
        if not self.spin0:
            w0 = None
        else:
            wigners0 = wpm_funcs.prepare_wigners(0,L1,L2,M1,M2,self.lmax)
            if not wigners0:
                w0 = None
            else:
                allowed_l, wigners0 = wigners0
                wlm_l = wpm_funcs.get_wlm_l(self.wlm,m,self.lmax,allowed_l)
                prefac = wpm_funcs.w_factor(allowed_l,L1,L2)
                w0 = (-1)**np.abs(M1) * np.sum(wigners0*prefac*wlm_l)
           
        if not self.spin2 or np.logical_or(L1 < 2, L2 < 2):
            wp, wm = None, None
        else:
            wigners2 = wpm_funcs.prepare_wigners(2,L1,L2,M1,M2,self.lmax)
            if not wigners2:
                wp, wm = None, None
            else:
                allowed_l,wp_l,wm_l = wigners2
                prefac = wpm_funcs.w_factor(allowed_l,L1,L2)
                wlm_l = wpm_funcs.get_wlm_l(self.wlm,m,self.lmax,allowed_l)
                wlm_l_large = np.where(np.abs((wlm_l)) > 1e-17,wlm_l,0)
                wp = 0.5 * (-1)**np.abs(M1) * np.sum(prefac*wlm_l*wp_l)
                wm = 0.5 * 1j * (-1)**np.abs(M1) * np.sum(prefac*wlm_l*wm_l)
            
        return (inds, w0, wp, wm)

    def save_w_element(self, result):
        inds, w0, wp, wm = result
        if w0:
            self.w0_arr[inds] = w0
        if wp or wm:
            inds = (slice(0,2),*inds)
            self.wpm_arr[inds] = [wp,wm]
    
    def calc_w_arrs(self,verbose=True):
        if not self.w0_arr and not self.wpm_arr:
            self.initiate_w_arrs()
        
        arglist = []
        if verbose:
            print('Preparing list of l, m arguments')
        for l1,L1 in enumerate(self.L):
            M1_arr = np.arange(-L1,L1+1)
            for l2,L2 in enumerate(self.L):
                M2_arr = np.arange(-L2,L2+1)
                
                for m1,M1 in enumerate(M1_arr):
                    for m2,M2 in enumerate(M2_arr):
                        arglist.append((L1,L2,M1,M2))
                        
        #n_proc = mup.cpu_count() - 1
        
        """ if verbose:
            print(f'Computing W_lmlpmp with {n_proc} cores')     """            
        
        #pool = mup.Pool(processes=n_proc)
        #with mup.Pool(processes=n_proc) as pool:
        for arg in arglist:
            
            result = self.calc_w_element(*arg)
            self.save_w_element(result)
                #pool.apply_async(self.calc_w_element, args=arg,callback=self.save_w_element)
            #pool.close()
            #pool.join()

        if self.spin0 and self.spin2:
            self.w_arr = np.append(self.wpm_arr,self.w0_arr,axis=0)
            self.w0_arr = None # could also delete these attributes from the instance itself to make space?
            self.wpm_arr = None

        elif self.spin0:
            helper = np.empty_like(self.w0_arr)[None,:,:,:,:]
            self.w_arr = np.append(np.append(helper,helper,axis=0),self.w0_arr,axis=0)

        elif self.spin2:
            self.w_arr = self.wpm_arr




                    
    

     

        

