#import axion_ppa
import priors
import axion_ppa
import numpy as np
import matplotlib.pyplot as plt
from PTMCMCSampler.PTMCMCSampler import PTSampler
from chainconsumer import ChainConsumer
import sys
import datetime
import os
    
PSR_DICT_LIST = axion_ppa.Load_Pulsars()
PSR_NAME_LIST = list(PSR_DICT_LIST.keys())

pulsars = [axion_ppa.Pulsar(PSR_DICT_LIST[psrn]) for psrn in PSR_DICT_LIST ]


"""
Get the prior range
"""

tag = "AF" ; sig0 = "Auto" ; sig1 = "Full"
lma_min = float(sys.argv[1])
lma_max = float(sys.argv[2])
dlnlike = float(sys.argv[3])


"""
Construct the array
"""

array = axion_ppa.Array(pulsars)
array.NOBS_TOTAL
ones = np.ones(array.NPSR)
zeros = np.zeros(array.NPSR)
lnlike_sig1_raw = array.Generate_Lnlike_Function( method=sig1 )
lnlike_sig0_raw = array.Generate_Lnlike_Function( method=sig0 )


NSS = np.sum( array.NSUBSETS_by_SS )
NPSR = array.NPSR


"""
Define likelihood and prior
"""
def Mapper(params):

    l10_EFAC = params[:NSS]
    l10_EQUAD = params[NSS:2*NSS]
    sDTE = params[2*NSS : 2*NSS+NPSR]
    v = params[2*NSS+NPSR ]
    l10_ma = params[2*NSS+NPSR + 1]
    l10_Sa = params[2*NSS+NPSR + 2]

    return l10_EFAC , l10_EQUAD , sDTE , v , l10_ma , l10_Sa

def lnlike_sig0( params ):
    lnlike_val = lnlike_sig0_raw(*Mapper(params))
    return lnlike_val

def lnlike_sig1( params ):
    lnlike_val = lnlike_sig1_raw(*Mapper(params))
    return lnlike_val



l10_EFAC_lp , l10_EFAC_sp = priors.gen_uniform_lnprior(-5,5)
l10_EQUAD_lp , l10_EQUAD_sp = priors.gen_uniform_lnprior(-8,2)
sDTE_lp = array.sDTE_LNPRIOR
v_lp , v_sp = priors.gen_uniform_lnprior(0,10)
l10_ma_lp , l10_ma_sp = priors.gen_uniform_lnprior(lma_min,lma_max)
l10_Sa_lp , l10_Sa_sp = priors.gen_uniform_lnprior(-8,2)

def lnprior_nonmodel( params ):
    l10_EFAC , l10_EQUAD , sDTE , v , l10_ma , l10_Sa =  Mapper( params )

    LP_1 = np.sum( [l10_EFAC_lp(l10_EFAC[i]) for i in range(NSS)] )
    LP_2 = np.sum( [l10_EQUAD_lp(l10_EQUAD[i]) for i in range(NSS)] )
    LP_3 = np.sum(  [sDTE_lp[i](sDTE[i]) for i in range(NPSR)]  )
    LP_4 = v_lp( v )
    LP_5 = l10_ma_lp( l10_ma )
    LP_6 = l10_Sa_lp( l10_Sa ) 
    #print(LP_1,LP_2,LP_3,LP_4,LP_5,LP_6)
    return  np.sum([LP_1,LP_2,LP_3,LP_4,LP_5,LP_6])





"""
Add hyper parameters
"""
def lnlike( all_params ):
    nmodel = all_params[0]
    if nmodel < 0:
        return lnlike_sig0( all_params[1:] ) + dlnlike
    elif nmodel >= 0:
        return lnlike_sig1( all_params[1:] ) 

nmodel_lp,nmodel_sp = priors.gen_uniform_lnprior(-1,1)
def lnprior( all_params ):
    lnprior_val = lnprior_nonmodel( all_params[1:] ) + nmodel_lp(all_params[0])
    return  lnprior_val


def get_init( ):
    init_val = [nmodel_sp()] +[l10_EFAC_sp() for i in range(NSS)] + [l10_EQUAD_sp() for i in range(NSS)] + ones.tolist() + [v_sp()] + [l10_ma_sp()] + [l10_Sa_sp()]
    return np.array(init_val)


        
"""
Prepare the run
"""

with open("chain_dir.txt",'r') as f:
    predir = f.read()+"/Signal_Chain/" 

now = datetime.datetime.now()
#now.strftime("%d-%m-%H:%M")

name = predir + tag + now.strftime("_%d_%m_%y")+ "/" + tag + f"_{lma_min:.2f}_{lma_max:.2f}_Np{NPSR}_Ns{NSS}_dL{dlnlike:.0f}"
os.system("mkdir -p "+name)

print(name)
#for info in ppa.info:
#    words.append(" ".join(info)+"\n")
#words.append("\n")
    
# with open(name+"/info.txt","a") as f:
#     f.writelines(words)

"""
Run the sampler
"""
init = get_init()

groups=None
cov = np.diag(np.ones(len(init)))
sampler = PTSampler( len(init) ,lnlike,lnprior,groups=groups,cov = cov,resume=False, outDir = name )
sampler.sample(np.array(init),500000,isave=100)




