# example: mpiexec -np 8 python run_spa.py red ionfr 10cm & mpiexec -np 8 python run_spa.py red noiono 10cm & mpiexec -np 8 python run_spa.py white ionfr 10cm & mpiexec -np 8 python run_spa.py white noiono 10cm &
# example: mpiexec -np 8 python run_spa.py red ionfr 20cm & mpiexec -np 8 python run_spa.py red noiono 20cm & mpiexec -np 8 python run_spa.py white ionfr 20cm & mpiexec -np 8 python run_spa.py white noiono 20cm &
import ppa
import numpy as np
from PTMCMCSampler.PTMCMCSampler import PTSampler
import json
import glob
import sys


burn = 250
Tmax = 10

def z_thermo(burn =0,outdir = "./"):
    files = sorted(glob.glob(outdir+"/chain*.txt"))[::-1]
    beta = []
    lnlmean = []
    for f in files:
        if "chain_1.txt" in f:
            pass
        else:
            temp = float(f.split("/")[-1].split("_")[-1][:-4])
            beta.append(1/temp)
            chain = np.loadtxt(f,skiprows=burn)
            lnlmean.append(np.mean(chain[:,-3]))
    beta = np.array(beta)
    sort = np.argsort(beta)
    beta = beta[sort]
    lnlmean = np.array(lnlmean)[sort]
    Z = np.sum((lnlmean[:-1] + lnlmean[1:])*np.diff(beta))/2
    return Z,beta

with open("chain_dir.txt",'r') as f:
    predir = f.read().split('\n')[0]+"/SPA_Chain/" 

PSR_DICT_LIST = ppa.Load_All_Pulsar_Info()
PSR_NAME_LIST = list(PSR_DICT_LIST.keys())

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

if sys.argv[1]=="white":

    for i,psrn in enumerate(PSR_NAME_LIST):  

        psr_p0 = ppa.Pulsar( PSR_DICT_LIST[ psrn ],det_order=0,iono=sys.argv[2],subset=sys.argv[3],nfreqs_dict={"ionfr_10cm":0 , "ionfr_20cm":0,"noiono_10cm":0,"noiono_20cm":0})
        psr_p1 = ppa.Pulsar( PSR_DICT_LIST[ psrn ],det_order=1,iono=sys.argv[2],subset=sys.argv[3],nfreqs_dict={"ionfr_10cm":0 , "ionfr_20cm":0,"noiono_10cm":0,"noiono_20cm":0})
        psr_p2 = ppa.Pulsar( PSR_DICT_LIST[ psrn ],det_order=2,iono=sys.argv[2],subset=sys.argv[3],nfreqs_dict={"ionfr_10cm":0 , "ionfr_20cm":0,"noiono_10cm":0,"noiono_20cm":0})
            
        array_p0 = ppa.Array([psr_p0])
        array_p1 = ppa.Array([psr_p1])
        array_p2 = ppa.Array([psr_p2])
            
        if array_p2.NPSR != 0:
            print(psrn)
            _lnlike_p0 = array_p0.Generate_Lnlike_Function(adm_signal='none')
            _lnlike_p1 = array_p1.Generate_Lnlike_Function(adm_signal='none')
            _lnlike_p2 = array_p2.Generate_Lnlike_Function(adm_signal='none')
            def lnlike(x):
                l10_EFAC , l10_EQUAD  = np.array(x)
                lnlval = _lnlike_p2( [l10_EFAC] , [l10_EQUAD] , [-8.] , [0.] ,1., -22. , -8. )
                return lnlval
                
            lnlike_ref_p0 = _lnlike_p0( [0.] , [-8.] , [-8.] , [0.] ,1., -22. , -8. )
            lnlike_ref_p1 = _lnlike_p1( [0.] , [-8.] , [-8.] , [0.] ,1., -22. , -8. )
            lnlike_ref_p2 = _lnlike_p2( [0.] , [-8.] , [-8.] , [0.] ,1., -22. , -8. )

            #l10_EFAC_lp , l10_EFAC_sp = ppa.gen_exp_lnprior(np.log10(0.99),np.log10(1.01))
            l10_EFAC_lp , l10_EFAC_sp = ppa.gen_uniform_lnprior(-2,2)
            l10_EQUAD_lp , l10_EQUAD_sp = ppa.gen_uniform_lnprior(-8,2)

            def lnprior( x ):
                l10_EFAC , l10_EQUAD  =  x
                LP_1 = l10_EFAC_lp(l10_EFAC)
                LP_2 = l10_EQUAD_lp(l10_EQUAD)
                return  np.sum([LP_1,LP_2])


            init = np.array([l10_EFAC_sp() , l10_EQUAD_sp()  ])
            outdir = predir+"/white_" + sys.argv[2] + "_" + psr_p2.PSR_NAME+"_" + sys.argv[3]
                
            sampler = PTSampler(len(init),lnlike,lnprior,cov = np.diag(np.ones(len(init)))*0.01,\
                                    resume=False,outDir=outdir,verbose=True)
            sampler.sample(init,1000000,writeHotChains=True,Tskip=2000,thin=1000,Tmax = Tmax)


elif sys.argv[1]=="red":
    for i,psrn in enumerate(PSR_NAME_LIST):  
        res = {}
        psr = ppa.Pulsar( PSR_DICT_LIST[ psrn ],det_order=2,iono=sys.argv[2],subset=sys.argv[3],\
            nfreqs_dict={"ionfr_10cm":30 , "ionfr_20cm":30,"noiono_10cm":30,"noiono_20cm":30})
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

            outdir = predir+"/red_" + sys.argv[2] + "_" + psr.PSR_NAME + "_" + sys.argv[3]

            sampler = PTSampler(len(init),lnlike,lnprior,cov = np.diag(np.ones(len(init)))*0.01,\
                                    resume=False,outDir=outdir,verbose=True)
            sampler.sample(init,1000000,writeHotChains=True,Tskip=2000,thin=1000,Tmax=Tmax)


