# example: python run_spa.py ionfr 10cm
import ppa
import numpy as np
import matplotlib.pyplot as plt
from PTMCMCSampler.PTMCMCSampler import PTSampler
import json
import glob
import sys
import corner


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

k = 0
logbf = {}
logz_white = {}

for i,psrn in enumerate(PSR_NAME_LIST):  

    psr_p0 = ppa.Pulsar( PSR_DICT_LIST[ psrn ],det_order=0,iono=sys.argv[1],subset=sys.argv[2],nfreqs_dict={"ionfr_10cm":0 , "ionfr_20cm":0,"noiono_10cm":0,"noiono_20cm":0})
    psr_p1 = ppa.Pulsar( PSR_DICT_LIST[ psrn ],det_order=1,iono=sys.argv[1],subset=sys.argv[2],nfreqs_dict={"ionfr_10cm":0 , "ionfr_20cm":0,"noiono_10cm":0,"noiono_20cm":0})
    psr_p2 = ppa.Pulsar( PSR_DICT_LIST[ psrn ],det_order=2,iono=sys.argv[1],subset=sys.argv[2],nfreqs_dict={"ionfr_10cm":0 , "ionfr_20cm":0,"noiono_10cm":0,"noiono_20cm":0})
        
    array_p0 = ppa.Array([psr_p0])
    array_p1 = ppa.Array([psr_p1])
    array_p2 = ppa.Array([psr_p2])
        
    if array_p2.NPSR != 0:
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

        l10_EFAC_lp , l10_EFAC_sp = ppa.gen_uniform_lnprior(-2,2)
        l10_EQUAD_lp , l10_EQUAD_sp = ppa.gen_uniform_lnprior(-8,2)

        def lnprior( x ):
            l10_EFAC , l10_EQUAD  =  x
            LP_1 = l10_EFAC_lp(l10_EFAC)
            LP_2 = l10_EQUAD_lp(l10_EQUAD)
            return  np.sum([LP_1,LP_2])


        init = np.array([l10_EFAC_sp() , l10_EQUAD_sp()  ])
        outdir = predir+"/white_" + sys.argv[1] + "_" + psr_p2.PSR_NAME+"_" + sys.argv[2]
            

        z,beta = z_thermo(burn=burn,outdir=outdir)
        logz_white.update({psrn:z})
        logbf.update({psrn: [ lnlike_ref_p1 - lnlike_ref_p0 , lnlike_ref_p2 - lnlike_ref_p1, z - np.trapz([lnlike_ref_p2]*len(beta)  , beta) ]  } )
        #print( psrn   ,"%.1f"%(logbf[psrn][0]) , "&%.1f"%(logbf[psrn][1])  , "&%.1f"%(logbf[psrn][2]) \
        #                ,"& %.2f"%np.log10(np.median(psr_p2.DPA_ERR[0])),"& %.2f"%np.log10(np.std(psr_p2.DPA[0])) )
            
    
#=====================================================#
#     M + EFAC + EQUAD + Sred + Gamma                 #
#=====================================================#
k = 0
kk = 0
string = sys.argv[1]+"_"+sys.argv[2]+" output"
print(len(string)*"*")
print(string)
print(len(string)*"*")

print("index & pulsar & log10EF & log10EQ/rad & log10Sr/rad & Gamma & logBF (0,1)/(0) & logBF (0,1,2)/(0,1) & logBF (0,1,2,w)/(0,1,2) & logBF (0,1,2,w,r)/(0,1,2,w)\\\\")
for i,psrn in enumerate(PSR_NAME_LIST):  
    res = {}
    psr = ppa.Pulsar( PSR_DICT_LIST[ psrn ],det_order=2,iono=sys.argv[1],subset=sys.argv[2],\
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

        outdir = predir+"/red_" + sys.argv[1] + "_" + psr.PSR_NAME + "_" + sys.argv[2]

        z,beta = z_thermo(burn=burn,outdir=outdir)
        logbfred = z - logz_white[psrn]
        chain = np.loadtxt(outdir+"/chain_1.0.txt",skiprows=burn)
            
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
        spa_results[psrn].update( { sys.argv[1]+"_"+sys.argv[2]:(med1,med2,med3,med4,(logbf[psrn][0],logbf[psrn][1],logbf[psrn][2],logbfred))}  )
            
        # corner plot
        corner.corner(chain[:,[0,1,2,3]],smooth=0,labels=[r"$\log_{10}$EF" , r"$\log_{10}$EQ" , r"$\log_{10}S_{\rm red}$",r"$\Gamma$"],show_titles=True,quiet=True);
        plt.suptitle(psrn+"_"+sys.argv[1]+"_"+sys.argv[2])
        plt.savefig("Figures/"+psrn+"_"+sys.argv[1]+"_"+sys.argv[2]+".jpg")
        plt.close()

        # tabulation
        #print(psrn,subset,"%.1f"%z,len(chain)) 
            
        print( kk+1,'&',psrn, "& $%.2f^{+%.2f}_{-%.2f}$  & $% .2f^{+%.2f}_{-%.2f} $ & $ % .2f^{+%.2f}_{-%.2f}$ & $% .2f^{+%.2f}_{-%.2f}$"%(med1,max1,min1,med2,max2,min2,med3,max3,min3,med4,max4,min4),\
                        "& %.1f"%(logbf[psrn][0])  , "& %.1f"%(logbf[psrn][1])  ,\
                        "& %.1f"%(logbf[psrn][2]) ,  "& %.1f"%(logbfred)+"\\\\" )
        kk+=1

with open("ppa/Parfile/spa_results.json",'w') as f:
    json.dump(spa_results,f,indent=2)
    print("ppa/Parfile/spa_results.json : spa result updated.")
