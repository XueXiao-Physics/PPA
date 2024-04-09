#import axion_ppa
import priors
import ppa
import numpy as np
from PTMCMCSampler.PTMCMCSampler import PTSampler
import sys
import datetime
import os
import json
import argparse
try:
    import mpi4py
except Exception:
    print("no mpi4py")



#=====================================================#
#    Read the argument                                #
#=====================================================#

parser = argparse.ArgumentParser(
    prog = "PPA",
    description="Run Bayesian Analysis of PPA"
)

parser.add_argument("-lma_min", action="store" , type=float , required=True )
parser.add_argument("-lma_max", action="store" , type=float , required=True )
parser.add_argument("-order" , action="store", type=int , default="2" )
parser.add_argument("-dlnprior" , action = "store" , type = float , default="0" )

parser.add_argument("-mock_method" , choices=["none" , "white" , "auto" ,"data"] , default="data")
parser.add_argument("-mock_lma" , action = "store" , type = float )
parser.add_argument("-mock_lSa" , action = "store" , type = float )
parser.add_argument("-pulsar"   , action = "store" , type = int )
parser.add_argument("-iono"     , choices=["None","Subt"] , required=True)
parser.add_argument("-subset"  , choices=["10cm","20cm","All"] , default = "All" , required=True)
parser.add_argument("-bf" , choices = ["af" , "na" , "nf"] , default = "af" , required=True)

args = parser.parse_args()

#=====================================================#
#    Load the pulsars                                 #
#=====================================================#
tag = args.mock_method
tag += f"_d{args.dlnprior:.0f}_O{args.order}_i" + args.iono + "_" + args.subset + "_"

PSR_DICT = ppa.Load_All_Pulsar_Info()

pulsars = [ ppa.Pulsar( PSR_DICT[psrn] , order = args.order , iono = args.iono , subset=args.subset ) for psrn in PSR_DICT ]


PSR_NAME_LIST = [psr.PSR_NAME for psr in pulsars]
if type(args.pulsar) == int : 
    
    pulsars = [pulsars[args.pulsar]]
    tag += pulsars[0].PSR_NAME + '_'

else:
    pulsars = pulsars
#print(len(pulsars))



#=====================================================#
#    Construct the array                              #
#=====================================================#


array = ppa.Array(pulsars)
if args.mock_method == "white":
    array.DPA = array.Gen_White_Mock_Data()
    print("Data replaced by mock data using the redefined measurement error.")
elif args.mock_method == "auto":
    array.DPA = array.Gen_Red_Mock_Data("auto",args.mock_lma , args.mock_lSa)
    print("Data replaced by mock data using the redefined measurement error.")
    print("Additional Red noise added: lma=%.1f, lSa=%.1f"%(args.mock_lma,args.mock_lSa), ". Method: " + "auto")
    tag += "_%.1f_%.1f"%(args.mock_lma,args.mock_lSa)
elif args.mock_method == "full":
    array.DPA = array.Gen_Red_Mock_Data("full",args.mock_lma , args.mock_lSa)
    print("Data replaced by mock data using the redefined measurement error.")
    print("Additional Red noise added: lma=%.1f, lSa=%.1f"%(args.mock_lma,args.mock_lSa), ". Method: " + "auto")
    tag += "_%.1f_%.1f"%(args.mock_lma,args.mock_lSa)
else:
    raise


ones = np.ones(array.NPSR)
zeros = np.zeros(array.NPSR)
if args.bf == "af":
    lnlike_sig0_raw = array.Generate_Lnlike_Function( method="Auto" )
    lnlike_sig1_raw = array.Generate_Lnlike_Function( method="Full" )
elif args.bf == "na":
    lnlike_sig0_raw = array.Generate_Lnlike_Function( method="None" )
    lnlike_sig1_raw = array.Generate_Lnlike_Function( method="Auto" )
elif args.bf == "nf":
    lnlike_sig0_raw = array.Generate_Lnlike_Function( method="None" )
    lnlike_sig1_raw = array.Generate_Lnlike_Function( method="Full" )
else:
    raise

NSS = np.sum( array.NSUBSETS_by_SS )
NPSR = array.NPSR

tag += args.bf + "_"
tag += f"Np{NPSR}"
#print(tag)


#=====================================================#
#    Mapping the parameters                           #
#=====================================================#
def Mapper(params):

    l10_EFAC = params[:NSS]
    l10_EQUAD = params[NSS:2*NSS]
    sDTE = params[2*NSS : 2*NSS+NPSR]
    l10_ma = params[2*NSS+NPSR ]
    l10_Sa = params[2*NSS+NPSR + 1]

    return l10_EFAC , l10_EQUAD ,  sDTE  , l10_ma , l10_Sa


#=====================================================#
#    Define lnlike                                    #
#=====================================================#

def lnlike_sig0( params ):
    lnlike_val = lnlike_sig0_raw(*Mapper(params))
    return lnlike_val

def lnlike_sig1( params ):
    lnlike_val = lnlike_sig1_raw(*Mapper(params))
    return lnlike_val




#=====================================================#
#    Define lnprior                                   #
#=====================================================#

l10_EFAC_lp , l10_EFAC_sp = priors.gen_uniform_lnprior(-1,3)
l10_EQUAD_lp , l10_EQUAD_sp = priors.gen_uniform_lnprior(-8,2)
sDTE_lp = array.sDTE_LNPRIOR
l10_ma_lp , l10_ma_sp = priors.gen_uniform_lnprior(args.lma_min,args.lma_max)
l10_Sa_lp , l10_Sa_sp = priors.gen_uniform_lnprior(-8,2)

def lnprior_nonmodel( params ):
    l10_EFAC , l10_EQUAD ,  sDTE , l10_ma , l10_Sa =  Mapper( params )

    LP_1 = np.sum( [l10_EFAC_lp(l10_EFAC[i]) for i in range(NSS)] )
    LP_2 = np.sum( [l10_EQUAD_lp(l10_EQUAD[i]) for i in range(NSS)] )
    LP_3 = np.sum(  [sDTE_lp[i](sDTE[i]) for i in range(NPSR)]  )
    LP_4 = l10_ma_lp( l10_ma )
    LP_5 = l10_Sa_lp( l10_Sa ) 
    #print(LP_1,LP_2,LP_3,LP_4,LP_5,LP_6)
    return  np.sum([LP_1,LP_2,LP_3,LP_4,LP_5])





#=====================================================#
#    Add hyper paramter nmodel                        #
#=====================================================#
def lnlike( all_params ):
    nmodel = all_params[0]
    if nmodel < 0:
        lnlike_val =  lnlike_sig0( all_params[1:] )
    elif nmodel >= 0:
        lnlike_val =  lnlike_sig1( all_params[1:] ) 
    return lnlike_val


if args.dlnprior == np.inf:
    add1 = 0
    add2 = -np.inf
    nmodel_lp,nmodel_sp = priors.gen_uniform_lnprior(-1,0)
elif args.dlnprior == -np.inf:
    add1 = -np.inf
    add2 = 0
    nmodel_lp,nmodel_sp = priors.gen_uniform_lnprior(0,1)
else:
    add1 = args.dlnprior
    add2 = 0
    nmodel_lp,nmodel_sp = priors.gen_uniform_lnprior(-1,1)

def lnprior( all_params ):
    
    nmodel = all_params[0] 
    lnprior_val = lnprior_nonmodel( all_params[1:] ) + nmodel_lp(all_params[0])
    if nmodel < 0:
        lnprior_val += add1
    elif nmodel >= 0:
        lnprior_val += add2
    return  lnprior_val


#=====================================================#
#    Initial values                                   #
#=====================================================#


def get_init():
    l10_EFAC_bf , l10_EQUAD_bf = array.Load_bestfit_params( )
    #l10_EFAC_bf += np.random.rand(len(l10_EFAC_bf))- 0.5
    #l10_EQUAD_bf += np.random.rand(len(l10_EQUAD_bf))- 0.5
    #K_bf += np.random.rand(len(K_bf)) * 0.001
    sDTE = np.random.rand(len(ones))*0.1 + 1 - 0.05
    

    init_val = [nmodel_sp()] +[l10_EFAC_sp() for i in range(NSS)] + [l10_EQUAD_sp() for i in range(NSS)]   + sDTE.tolist()  + [l10_ma_sp()] + [l10_Sa_sp()]
    #init_val = [nmodel_sp()] + l10_EFAC_bf.tolist() + l10_EQUAD_bf.tolist() + K_bf.tolist() + sDTE.tolist() + [v_sp()] + [l10_ma_sp()] + [l10_Sa_sp()]
    return np.array(init_val)
        




#=====================================================#
#    Prepare the run                                  #
#=====================================================#

with open("chain_dir.txt",'r') as f:
    predir = f.read().split('\n')[0]+"/Signal_Chain/" 

now = datetime.datetime.now()
#now.strftime("%d-%m-%H:%M")

#name = predir + tag + now.strftime("_T_%d_%m_%y")+ "/" + tag + f"_{dlnlike:.0f}_{lma_min:.2f}_{lma_max:.2f}_Np{NPSR}_Ns{NSS}"
name = predir + tag +  f"/bin_{args.lma_min:.2f}_{args.lma_max:.2f}"

#os.system("mkdir -p "+name)


#=====================================================#
#    Run the sampler                                  #
#=====================================================#
init = get_init()
print( lnlike(init))

groups = [np.arange(len(init)) , [0, 2*NSS+NPSR+1 , 2*NSS+NPSR + 2  ]]

iss = 1
for ipsr in range(NPSR):
    g = []
    nss = len( array.SUBSETS[ipsr] )
    g += np.arange(iss,iss+nss).tolist()
    g += np.arange(iss+NSS,iss+nss+NSS).tolist()
    g += [1+2*NSS+ipsr]
    iss += nss
    groups.append(g)


#groups += [[i+1,i+1+NSS] for i in range(NSS)]
cov = np.diag(np.ones(len(init)))
sampler = PTSampler( len(init) ,lnlike,lnprior,groups=groups,cov = cov,resume=False, outDir = name )
sampler.sample(np.array(init),5000000,thin=100)




