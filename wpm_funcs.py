import wigner 
import healpy as hp
import numpy as np


# what if the mask is a class containing information about spins, geometry and the wlmlpmp?

def w_factor(l,l1,l2):
    return np.sqrt((2*l1+1)*(2*l2+1)*(2*l+1)/(4*np.pi))


def prepare_wigners(spin,L1,L2,M1,M2,lmax):
    # prepare an array of wigners (or products thereof) of allowed l to be summed for given m1,m2,l1,l2
    # should return two arrays for wpm and one for w0
    m = M1 - M2
    w1lmin,w1lmax,w1cof = wigner.wigner_3jj(L1,L2,-M1,M2)
    if spin == 0:
        w0lmin,w0lmax,w0cof = wigner.wigner_3jj(L1,L2,0,0)
        allowed_l = np.arange(max(w0lmin,w1lmin),min(w0lmax,w1lmax,lmax)+1,dtype=int)
        if len(allowed_l) == 0:
            return None
        else:
            w0cof = w0cof[max(0,allowed_l[0]-int(w0lmin)):(-(int(w0lmax)-allowed_l[-1]) if (int(w0lmax)-allowed_l[-1]) > 0 else None)]
            w1cof = w1cof[max(0,allowed_l[0]-int(w1lmin)):(-(int(w1lmax)-allowed_l[-1]) if (int(w1lmax)-allowed_l[-1]) > 0 else None)]
            return allowed_l, w0cof*w1cof  

    elif spin == 2:
        w2plmin,w2plmax,w2pcof = wigner.wigner_3jj(L1,L2,2,-2)           
        w2mlmin,w2mlmax,w2mcof = wigner.wigner_3jj(L1,L2,-2,2)
        allowed_s2l = np.arange(min(w2plmin,w2mlmin),max(w2plmax,w2mlmax)+1)
        allowed_l = np.arange(max(allowed_s2l[0],w1lmin),min(allowed_s2l[-1],w1lmax,lmax)+1,dtype=int)
        if len(allowed_l) == 0 or np.abs(m) > lmax:
            return None
        else:
            w1cof_2 = w1cof[max(0,allowed_l[0]-int(w1lmin)):(-(int(w1lmax)-allowed_l[-1]) if (int(w1lmax)-allowed_l[-1]) > 0 else None)]
            w2pcof = w2pcof[max(0,allowed_l[0]-int(w2plmin)):(-(int(w2plmax)-allowed_l[-1]) if (int(w2plmax)-allowed_l[-1]) > 0 else None)]
            w2mcof = w2mcof[max(0,allowed_l[0]-int(w2mlmin)):(-(int(w2mlmax)-allowed_l[-1]) if (int(w2mlmax)-allowed_l[-1]) > 0 else None)]
            w2sum = w2pcof + w2mcof
            w2diff = w2pcof - w2mcof          
            wp_l = w2sum * w1cof_2
            wm_l = w2diff * w1cof_2
            return allowed_l,wp_l,wm_l
        
    else:
        raise RuntimeError('Wigner 3j-symbols can only be calculated for spin 0 or 2 fields.')


def get_wlm_l(wlm,m,lmax,allowed_l):
    if m < 0:
        wlm_l = (-1) ** -m * np.conj(wlm[hp.sphtfunc.Alm.getidx(lmax, allowed_l, -m)])
    else: 
        wlm_l = wlm[hp.sphtfunc.Alm.getidx(lmax, allowed_l, m)]

    return wlm_l





