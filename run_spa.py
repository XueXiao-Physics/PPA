#import ppa
import ppa
import mpi4py
import numpy as np
import matplotlib.pyplot as plt
from PTMCMCSampler.PTMCMCSampler import PTSampler
import json
import scipy.stats as ss
import os
import glob
import sys

burn = 5000
Tmax = 10

def z_thermo(burn =0,outdir = "./"):
    files = sorted(glob.glob(outdir+"/chain*.txt"))[::-1]
    beta = []
    lnlmean = []
    for f in files:
        temp = float(f.split("/")[-1].split("_")[-1][:-4])
        beta.append(1/temp)
        chain = np.loadtxt(f,skiprows=burn)
        lnlmean.append(np.mean(chain[:,-3]))
    beta = np.array(beta)
    sort = np.argsort(beta)
    beta = beta[sort]
    lnlmean = np.array(lnlmean)[sort]
    Z = np.sum((lnlmean[:-1] + lnlmean[1:])*np.diff(beta))/2
    return Z

with open("chain_dir.txt",'r') as f:
    predir = f.read().split('\n')[0]+"/SPA_Chain/" 
PSR_DICT_LIST = ppa.Load_All_Pulsar_Info()
PSR_NAME_LIST = list(PSR_DICT_LIST.keys())
PSR_DICT_LIST

try:
    with open("ppa/Parfile/spa_results.json",'r') as f:
        spa_results = json.load(f)
except:
    spa_results = {}
    for psrn in PSR_NAME_LIST:
        spa_results.update({psrn:{}})

#=====================================================#
#     M + EFAC + EQUAD                                #
#=====================================================#

k = 0
logz_all = []
for i,psrn in enumerate(PSR_NAME_LIST):  
    for subset in ["10cm"]:#["10cm","20cm"]:
        
        psr = ppa.Pulsar( PSR_DICT_LIST[ psrn ],order=2,iono=sys.argv[2],subset=subset,nfreqs_dict={"ionfr_10cm":0 , "ionfr_20cm":0,"noiono_10cm":0,"noiono_20cm":0})
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
            

            outdir = predir+"/white_"+sys.argv[2]+"_"+psr.PSR_NAME+"_"+subset
            if sys.argv[1]=="read":
                z = z_thermo(burn=burn,outdir=outdir)
                logz_all.append(z)
                print(psrn,subset,"%.1f"%z)
                del(z)
            elif sys.argv[1] == "white":
                print(psrn,subset)
                sampler = PTSampler(len(init),lnlike,lnprior,cov = np.diag(np.ones(len(init)))*0.01,\
                                    resume=True,outDir=outdir,verbose=True)
                sampler.sample(init,100000,writeHotChains=True,Tmax = Tmax)
    
#=====================================================#
#     M + EFAC + EQUAD + Sred + Gamma                 #
#=====================================================#
k = 0
for i,psrn in enumerate(PSR_NAME_LIST):  
    res = {}
    for subset in ["10cm"]:#["10cm","20cm"]:
        psr = ppa.Pulsar( PSR_DICT_LIST[ psrn ],order=2,iono=sys.argv[2],subset=subset,nfreqs_dict={"ionfr_10cm":30 , "ionfr_20cm":30,"noiono_10cm":30,"noiono_20cm":30})
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

            outdir = predir+"/red30_"+sys.argv[2]+"_"+psr.PSR_NAME+"_"+subset
            if sys.argv[1]=="read":
                z = z_thermo(burn=burn,outdir=outdir)
                logbf = z - logz_all[k]
                chain = np.loadtxt(outdir+"/chain_1.0.txt")
                
                med1 = np.median( chain[:,0])
                min1 = med1 - np.quantile( chain[:,0],0.16 )
                max1 = np.quantile( chain[:,0],0.84 ) - med1 


                med2 = np.median( chain[:,1])
                min2 = med2 - np.quantile( chain[:,1],0.16 )
                max2 = np.quantile( chain[:,1],0.84 ) - med2


                med3 = np.median( chain[:,2])
                min3 = med3 - np.quantile( chain[:,2],0.16 )
                max3 = np.quantile( chain[:,2],0.84 ) - med3


                med4 = np.median( chain[:,3])
                min4 = med4 - np.quantile( chain[:,3],0.16 )
                max4 = np.quantile( chain[:,3],0.84 ) - med4
                spa_results[psrn].update( { sys.argv[2]+"_"+subset:(med1,med2,med3,med4,(logbf))}  )


                print( k+1,'&',psrn,"& %.2f"%np.log10(np.median(psr.DPA_ERR[0])),"& %.2f"%np.log10(np.std(psr.DPA[0])) ,\
                  "& $%.2f^{+%.2f}_{-%.2f}$  & $% .2f^{+%.2f}_{-%.2f} $ & $ % .2f^{+%.2f}_{-%.2f}$ & $% .2f^{+%.2f}_{-%.2f}$"%(med1,max1,min1,med2,max2,min2,med3,max3,min3,med4,max4,min4),\
                    "& %.1f"%(logbf)+"\\\\" )
                del(z,logbf)
            elif sys.argv[1]=="red":
                print(psrn,subset)
                sampler = PTSampler(len(init),lnlike,lnprior,cov = np.diag(np.ones(len(init)))*0.01,\
                                    resume=True,outDir=outdir,verbose=True)
                sampler.sample(init,100000,writeHotChains=True,Tmax=Tmax)
            k += 1

if sys.argv[1]=='read':
    with open("ppa/Parfile/spa_results.json",'w') as f:
        json.dump(spa_results,f,indent=2)
