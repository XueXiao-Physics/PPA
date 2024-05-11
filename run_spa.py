#import ppa
import ppa
import mpi4py
import numpy as np
import matplotlib.pyplot as plt
from PTMCMCSampler.PTMCMCSampler import PTSampler
from chainconsumer import ChainConsumer
from num2tex import num2tex
import json
import scipy.stats as ss
import corner
import os
    
PSR_DICT_LIST = ppa.Load_All_Pulsar_Info()
PSR_NAME_LIST = list(PSR_DICT_LIST.keys())
PSR_DICT_LIST

#=====================================================#
#     M + EFAC + EQUAD                                #
#=====================================================#
k = 0
logw_all = []
for i,psrn in enumerate(PSR_NAME_LIST):  
    for subset in ["10cm","20cm"]:
        
        psr = ppa.Pulsar( PSR_DICT_LIST[ psrn ],order=2,iono="ionfr",subset=subset,nfreqs_dict={"10cm":0 , "20cm":0})
        array = ppa.Array([psr])
        if array.NPSR != 0:
            _lnlike = array.Generate_Lnlike_Function(adm_signal='none')

            def lnlike(x):
                l10_EFAC , l10_EQUAD  = np.array(x)
                return _lnlike( [l10_EFAC] , [l10_EQUAD] , [-3.] , [0.] ,1., -22. , -8. )

            l10_EFAC_lp , l10_EFAC_sp = ppa.gen_uniform_lnprior(-2,2)
            l10_EQUAD_lp , l10_EQUAD_sp = ppa.gen_uniform_lnprior(-8,2)

            def lnprior( x ):
                l10_EFAC , l10_EQUAD  =  x

                LP_1 = l10_EFAC_lp(l10_EFAC)
                LP_2 = l10_EQUAD_lp(l10_EQUAD)

                return  np.sum([LP_1,LP_2])


            init = np.array([l10_EFAC_sp() , l10_EQUAD_sp()  ])
            

            outdir = "SPA_Chain/white_"+psr.PSR_NAME+"_"+subset
            if os.path.isdir(outdir):
                pass
            else:
                print(psrn,subset)
                sampler = PTSampler(len(init),lnlike,lnprior,cov = np.diag(np.ones(len(init)))*0.01,\
                                    resume=False,outDir=outdir,verbose=True)
                sampler.sample(init,100000)


#=====================================================#
#     M + EFAC + EQUAD + Sred + Gamma                 #
#=====================================================#

Res = {}
k = 0
for i,psrn in enumerate(PSR_NAME_LIST):  
    res = {}
    for subset in ["10cm","20cm"]:
        psr = ppa.Pulsar( PSR_DICT_LIST[ psrn ],order=2,iono="ionfr",subset=subset,nfreqs_dict={"10cm":30 , "20cm":30})
        array = ppa.Array([psr])
        if array.NPSR != 0:
            _lnlike = array.Generate_Lnlike_Function(adm_signal='none')

            def lnlike(x):
                l10_EFAC , l10_EQUAD , l10_S0red  ,Gamma = np.array(x)
                return _lnlike( [l10_EFAC] , [l10_EQUAD] , [l10_S0red] , [Gamma] ,1, -22 , -8 )

            l10_EFAC_lp , l10_EFAC_sp = ppa.gen_uniform_lnprior(-2,2)
            l10_EQUAD_lp , l10_EQUAD_sp = ppa.gen_uniform_lnprior(-8,2)
            l10_S0red_lp , l10_S0red_sp = ppa.gen_uniform_lnprior(-8,2)
            Gamma_lp    , Gamma_sp = ppa.gen_uniform_lnprior(-8,2)
            
            def lnprior( x ):
                l10_EFAC , l10_EQUAD , l10_S0red , Gamma =  x

                LP_1 = l10_EFAC_lp(l10_EFAC)
                LP_2 = l10_EQUAD_lp(l10_EQUAD)
                LP_3 = l10_S0red_lp(l10_S0red)
                LP_4 = Gamma_lp(Gamma)

                return  np.sum([LP_1,LP_2,LP_3,LP_4])

            

            init = np.array([l10_EFAC_sp() , l10_EQUAD_sp() , l10_S0red_sp() , Gamma_sp() ])

            outdir = "SPA_Chain/red30_"+psr.PSR_NAME+"_"+subset
            if os.path.isdir(outdir):
                pass
            else:
                print(psrn,subset)
                sampler = PTSampler(len(init),lnlike,lnprior,cov = np.diag(np.ones(len(init)))*0.01,\
                                    resume=False,outDir=outdir,verbose=True)
                sampler.sample(init,100000)
