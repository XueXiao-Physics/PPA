import ppa
import numpy as np
from PTMCMCSampler.PTMCMCSampler import PTSampler
import datetime
import argparse
from mpi4py import MPI
import sys
import json
with open("ppa/Parfile/spa_results.json",'r') as f:
    spa_results = json.load(f)



#=====================================================#
#    Read the argument                                #
#=====================================================#

parser = argparse.ArgumentParser(description="Run one simulation with specified parameters.")

# Argument for minimum log mass
parser.add_argument("-lma_min", action="store", type=float, default="-23.6",
                    help="Minimum log mass (default: -23.6)")

# Argument for maximum log mass
parser.add_argument("-lma_max", action="store", type=float, default="-18.5",
                    help="Maximum log mass (default: -18.5)")

# Argument for the det_order of the pulsar
parser.add_argument("-det_order", action="store", type=int, default="2",
                    help="det_order of the pulsar (default: 2)")

# Argument for the delta log prior
parser.add_argument("-dlnprior", action="store", type=float, default="0",
                    help="Delta log prior (default: 0)")

# Argument for the number of samples
parser.add_argument("-nsamp", action="store", type=int, default=5000000,
                    help="Number of samples (default: 5000000)")

# Arguments for mock data
parser.add_argument("-if_mock", action="store", choices=["True", "False"], default="False",
                    help="Flag to indicate if mock data should be used (default: False)")

parser.add_argument("-mock_noise", action="store", choices=["white", "red"], default="red",
                    help="Type of noise for mock data (default: red)")

parser.add_argument("-mock_adm", action="store", choices=["none", "auto", "full"], default="none",
                    help="Type of mock admittance (default: none)")

parser.add_argument("-mock_lma", action="store", type=float, default="-22.0",
                    help="Log mass for mock data (default: -22.0)")

parser.add_argument("-mock_lSa", action="store", type=float, default="-2.5",
                    help="Log Sa for mock data (default: -2.5)")

parser.add_argument("-mock_seed", action="store", type=int, default="10",
                    help="Seed for mock data generation (default: 10)")

# Arguments for pulsar data
parser.add_argument("-pulsar", action="store", type=int, default="-1",
                    help="Pulsar index (default: -1)")

parser.add_argument("-iono", action="store", choices=["ionfr", "noiono"], default='ionfr',
                    help="Ionospheric correction option (default: ionfr)")

parser.add_argument("-subset", action="store", choices=["10cm", "20cm", "all"], default="10cm",
                    help="Data subset to use (default: 10cm)")

parser.add_argument("-model", action="store", choices=["af", "na", "nf", "aa", "nn", "ff"], default="af",
                    help="Model type (default: af)")

parser.add_argument("-nfreqs", action="store", type=int, default="-1",
                    help="Number of frequencies (default: -1)")

parser.add_argument("-white", action="store", choices=["vary", "bestfit", "none"], default="vary",
                    help="White noise option (default: vary)")

# Parse the arguments
args = parser.parse_args()

#=====================================================#
#    Load the pulsars                                 #
#=====================================================#

tag_ana = f"_ana_d{args.dlnprior:.0f}_o{args.det_order}_w{args.white}_r{args.nfreqs}_" + args.iono  + "_" + args.subset + "_" + args.model + "_"

PSR_DICT = ppa.Load_All_Pulsar_Info()
PSR_NAMES_ALL = sorted(PSR_DICT.keys())
if args.pulsar >= 0:
    PSR_NAMES_SEL = [PSR_NAMES_ALL[args.pulsar]]
    tag_ana += PSR_NAMES_ALL[args.pulsar] + '_'
else:
    PSR_NAMES_SEL = PSR_NAMES_ALL
    select = np.arange(len(PSR_DICT))


pulsars=[]
for psrn in PSR_NAMES_SEL:
    nfreqs_dict_psr = {}  
    white_noise_dict_psr = {}
    for key in spa_results[psrn].keys():
        psr_noise = spa_results[psrn][key]
        lbf1,lbf2,lbf3,lbf4 = psr_noise[4]
        
        if args.white == "vary":
            white_noise_dict_psr.update( { key : "vary" } )
        elif args.white == "bestfit":
            white_noise_dict_psr.update( { key : [ psr_noise[0] , psr_noise[1] ] } )
        elif args.white == "none":
            white_noise_dict_psr.update( { key : [ 0 , -99 ] } ) 

        if args.nfreqs==-1:#
            if  lbf4 > 2.3:
                if psr_noise[3] <= -3:#
                    nfreqs_dict_psr.update( { key : 5 } )
                elif psr_noise[3] > -3:
                    nfreqs_dict_psr.update( { key : 30 } )
            else:
                nfreqs_dict_psr.update( { key : 0 } )
        else:
            nfreqs_dict_psr.update( { key : args.nfreqs } )
    pulsar = ppa.Pulsar(PSR_DICT[psrn],det_order = args.det_order \
                        , iono = args.iono , subset=args.subset
                        , nfreqs_dict = nfreqs_dict_psr , white_noise_dict = white_noise_dict_psr )
    print(psrn,args.iono,args.subset,args.white,args.nfreqs,[len(k) for k in pulsar.FREQS])
    pulsars.append(pulsar)


PSR_NAME_LIST = [psr.PSR_NAME for psr in pulsars]
array = ppa.Array(pulsars)
NSS = np.sum( array.NSUBSETS_by_SS )
NPSR = array.NPSR
if NSS ==0:
    print("No usable pulsar.")
    sys.exit()
tag_ana += f"Np{NPSR}" 


#=====================================================#
#    Mock data, also load pulsars                     #
#=====================================================#
if args.if_mock =="True":
    if args.mock_noise =="white":
        pulsars_mock = [ ppa.Pulsar( PSR_DICT[psrn] , det_order = args.det_order \
                       , iono = args.iono , subset = args.subset \
                       , nfreqs_dict={"ionfr_10cm":0,"ionfr_20cm":0 , "noiono_10cm":0 , "noiono_20cm":0} ) for psrn in PSR_DICT ]
        
    elif args.mock_noise =="red":
        # for mock data we consider more red noise bins, to test the efficiency of the subtraction.
        pulsars_mock = []
        for psrn in PSR_DICT:
            nfreqs_dict_psr = {}
            for key in spa_results[psrn].keys():
                psr_noise = spa_results[psrn][key]
                lbf1,lbf2,lbf3,lbf4 = psr_noise[4]
                if lbf4 > 2.3 and psr_noise[3]<=-3:
                    nfreqs_dict_psr.update( { key : 30 } )
                elif lbf4 > 2.3 and psr_noise[3]>=-3:
                    nfreqs_dict_psr.update( { key : 100 } )
                else:
                    nfreqs_dict_psr.update( { key : 0 } )

            pulsars_mock.append(ppa.Pulsar(PSR_DICT[psrn],det_order = args.det_order \
                                , iono = args.iono , subset=args.subset,nfreqs_dict=nfreqs_dict_psr ))

    PSR_NAME_LIST = [psr.PSR_NAME for psr in pulsars_mock]
    if args.pulsar >= 0 : 
        pulsars_mock = [pulsars_mock[args.pulsar]]
    elif args.pulsar == -1:
        pulsars_mock = pulsars_mock

    array_mock = ppa.Array(pulsars_mock)

    # important: replace the DPA of the original array
    array.DPA = array_mock.Gen_Mock_Data( noise_type=args.mock_noise , adm_signal=args.mock_adm \
                                     ,mock_lma=args.mock_lma ,mock_lSa=args.mock_lSa  , seed=args.mock_seed)
    if args.mock_adm =="none":
        tag = "mock_"+args.mock_noise+"_se%04d"%(args.mock_seed) + tag_ana
    else:
        tag = "mock_" + args.mock_noise + "_" + args.mock_adm \
            + "_%.2f_%.2f_se%i"%(args.mock_lma,args.mock_lSa,args.mock_seed) \
                + tag_ana
        
elif args.if_mock == "False":
    tag = "data" + tag_ana
else:
    raise


#=====================================================#
#    Construct the array                              #
#=====================================================#

ones = np.ones(array.NPSR)
zeros = np.zeros(array.NPSR)
model_mapper = {"n":"none" , "a":"auto" , "f":"full"}
lnlike_sig0_raw = array.Generate_Lnlike_Function( model_mapper[args.model[0]] )
lnlike_sig1_raw = array.Generate_Lnlike_Function( model_mapper[args.model[1]] )


#=====================================================#
#    Mapping the parameters                           #
#=====================================================#
def Mapper(params):

    l10_EFAC = params[:NSS]
    l10_EQUAD = params[NSS:2*NSS]
    l10_S0red = params[2*NSS:3*NSS]
    Gamma = params[3*NSS:4*NSS]
    sDTE = params[4*NSS : 4*NSS+NPSR]
    l10_ma = params[4*NSS+NPSR ]
    l10_Sa = params[4*NSS+NPSR + 1]

    return l10_EFAC , l10_EQUAD , l10_S0red , Gamma , sDTE  , l10_ma , l10_Sa


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

l10_EFAC_lp , l10_EFAC_sp = ppa.gen_uniform_lnprior(-2,2)
l10_EQUAD_lp , l10_EQUAD_sp = ppa.gen_uniform_lnprior(-8,2)
l10_S0red_lp , l10_S0red_sp = ppa.gen_uniform_lnprior(-8,2)
Gamma_lp    , Gamma_sp = ppa.gen_uniform_lnprior(-8,2)
sDTE_lp = array.sDTE_LNPRIOR
l10_ma_lp , l10_ma_sp = ppa.gen_uniform_lnprior(args.lma_min,args.lma_max)
l10_Sa_lp , l10_Sa_sp = ppa.gen_uniform_lnprior(-8,2)

def lnprior_nonmodel( params ):
    l10_EFAC , l10_EQUAD , l10_S0red , Gamma,  sDTE , l10_ma , l10_Sa =  Mapper( params )

    LP_1 = np.sum( [l10_EFAC_lp(l10_EFAC[i]) for i in range(NSS)] )
    LP_2 = np.sum( [l10_EQUAD_lp(l10_EQUAD[i]) for i in range(NSS)] )
    LP_3 = np.sum( [l10_S0red_lp(l10_S0red[i]) for i in range(NSS)] )
    LP_4 = np.sum( [Gamma_lp(Gamma[i]) for i in range(NSS)] )

    LP_5 = np.sum(  [sDTE_lp[i](sDTE[i]) for i in range(NPSR)]  )
    LP_6 = l10_ma_lp( l10_ma )
    LP_7 = l10_Sa_lp( l10_Sa ) 

    return  np.sum([LP_1,LP_2,LP_3,LP_4,LP_5,LP_6,LP_7])





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
    nmodel_lp,nmodel_sp = ppa.gen_uniform_lnprior(-1,0)
elif args.dlnprior == -np.inf:
    add1 = -np.inf
    add2 = 0
    nmodel_lp,nmodel_sp = ppa.gen_uniform_lnprior(0,1)
else:
    add1 = args.dlnprior
    add2 = 0
    nmodel_lp,nmodel_sp = ppa.gen_uniform_lnprior(-1,1)

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

    #l10_EFAC_bf , l10_EQUAD_bf , l10_S0red_bf , Gamma_bf = array.Load_bestfit_params( )
    sDTE = np.random.rand(len(ones))*0.1 + 1 - 0.05
    init_val = [nmodel_sp()] +[l10_EFAC_sp() for i in range(NSS)]\
                             + [l10_EQUAD_sp() for i in range(NSS)]\
                             + [l10_S0red_sp() for i in range(NSS)]\
                             + [Gamma_sp() for i in range(NSS)]\
                             + sDTE.tolist()\
                             + [l10_ma_sp()]\
                             + [l10_Sa_sp()]
    return np.array(init_val)
        


#=====================================================#
#    Prepare the run                                  #
#=====================================================#

with open("chain_dir.txt",'r') as f:
    predir = f.read().split('\n')[0]+"/Signal_Chain/" 

now = datetime.datetime.now()
name = predir + tag +  f"/bin_{args.lma_min:.2f}_{args.lma_max:.2f}"

#=====================================================#
#    Run the sampler                                  #
#=====================================================#
init = get_init()
#print( lnlike(init))

groups = [np.arange(len(init)) , [0, 4*NSS+NPSR+1 , 4*NSS+NPSR + 2  ]]

iss = 1
for ipsr in range(NPSR):
    g = []
    nss = len( array.SUBSETS[ipsr] )
    g += np.arange(iss,iss+nss).tolist()
    g += np.arange(iss+NSS,iss+nss+NSS).tolist()
    g += np.arange(iss+2*NSS,iss+nss+2*NSS).tolist()
    g += np.arange(iss+3*NSS,iss+nss+3*NSS).tolist()
    g += [1+4*NSS+ipsr]
    iss += nss
    groups.append(g)

cov = np.diag(np.ones(len(init)))



sampler = PTSampler( len(init) ,lnlike,lnprior,\
        groups=groups,cov = cov,resume = True, outDir = name,verbose=True )
sampler.sample(np.array(init),args.nsamp,\
        thin=1000,Tmax=20,Tskip=50000,writeHotChains=True)



