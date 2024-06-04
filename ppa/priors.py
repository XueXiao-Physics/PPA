import numpy as np
import astropy.coordinates as ac
from astropy import units as u
import scipy.integrate as si


"""
the function to calcuate sky location of the pulsars
"""
def get_psr_loc( ra_str , dec_str ):

    sky_loc0 = ac.SkyCoord(ra = ra_str , dec = dec_str , distance = 1 , unit= (u.hourangle, u.deg, u.kpc))
    sky_loc0 = np.array([sky_loc0.cartesian.x.value,sky_loc0.cartesian.y.value,sky_loc0.cartesian.z.value])
    
    return sky_loc0





"""
gen Half Gaussian prior
"""
def gen_DM_dist_lnprior(  ):

    def lnprior_noconst(x):
        if x <= 0.0:
            lnprior = -np.inf

        elif x > 0.0 and x <= 0.8:
            lnprior = -(x-0.8)**2 / 2 / 0.2**2

        elif x > 0.8 and x <= 1.2 :
            lnprior = 0.

        elif x >  1.2 :
            lnprior = -(x-1.2)**2 / 2 / 0.2**2
        
        return lnprior 
        
    prior = lambda x : np.exp(lnprior_noconst(x))
    const = si.quad( prior , 0 , 10 )[0] 
    lnprior = lambda x : lnprior_noconst(x) - np.log(const)

    return lnprior





def gen_PX_dist_lnprior( PX, PXerr ):
    D0 = 1 / PX
    Derr = D0 * PXerr

    def lnprior_noconst(x):

        if x <= 0:
            lnprior = -np.inf
        
        elif x > 0:
            lnprior = -0.5*( 1/x - 1 )**2 / Derr**2 - 2 * np.log(x)

        return lnprior
        
    prior = lambda x : np.exp(lnprior_noconst(x))
    const = si.quad( prior , 0 , 10 )[0]     
    lnprior = lambda x : lnprior_noconst(x) - np.log(const)

    return lnprior


def get_DTE(  DTE_DM ,  PX , PX_ERR ):

    # using DM distance
    if np.isnan(PX) or np.isnan(PX_ERR) or PX_ERR > 0.3*PX:
        lnprior = gen_DM_dist_lnprior()
        D0 = DTE_DM
    # using PX distance
    else:
        lnprior = gen_PX_dist_lnprior(PX,PX_ERR)
        D0 = 1/PX


    return D0 , lnprior



def gen_uniform_lnprior( a , b ):

    norm = b - a

    def lnprior( x ):
        if x <= a:
            lnp = -np.inf
        elif x > a and x <= b: 
            lnp = -np.log(norm)
        elif x > b:
            lnp = -np.inf
        return lnp
    
    def sampler():
        np.random.seed()
        return np.random.rand()*(b-a) + a
    
    return lnprior, sampler


def gen_exp_lnprior(a,b):

    norm = np.log(10) / (10**b - 10**a)

    def lnprior( x ):
        if x <= a:
            lnp = -np.inf
        elif x > a and x <= b:
            lnp = x * np.log(10) + np.log(norm) 
        elif x > b:
            lnp = -np.inf
        return lnp

    def sampler():
        np.random.seed()
        return np.log10( np.random.rand()*(10**b-10**a) + 10**a )

    return lnprior, sampler
