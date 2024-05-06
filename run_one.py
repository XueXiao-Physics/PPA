import ppa
import numpy as np
from PTMCMCSampler.PTMCMCSampler import PTSampler
import datetime
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
# For search
parser.add_argument("-lma_min"  , action="store" ,   type = float , default="-23.6"  )
parser.add_argument("-lma_max"  , action="store" ,   type = float , default="-18.5" )
parser.add_argument("-order"    , action="store",    type = int , default="2" )
parser.add_argument("-dlnprior" , action = "store" , type = float , default="0" )
parser.add_argument("-nsamp"    , action = "store" , type = int , default = 5000000)

# for mock data
parser.add_argument("-mock_method" , action = "store", choices=["none" , "white" , "auto" ,"full","data"] , default="data")
parser.add_argument("-mock_lma"    , action = "store" , type = float , default = "-22.0"  )
parser.add_argument("-mock_lSa"    , action = "store" , type = float , default = "-2.5" )
parser.add_argument("-mock_seed"   , action = "store" , type = int , default = "10" )



parser.add_argument("-pulsar"      , action = "store" , type = int , default="-1" )
parser.add_argument("-iono"        , action = "store", choices=["none","subt"] , default='subt')
parser.add_argument("-subset"      , action = "store", choices=["10cm","20cm","all"] , default = "all" )
parser.add_argument("-model"        , action = "store", choices = ["af" , "na" , "nf","a","n","f"] , default = "af" )
parser.add_argument("-nfreqs"       , action = "store", type = int , default="0" )

args = parser.parse_args()

#=====================================================#
#    Load the pulsars                                 #
#=====================================================#

tag_ana = f"_ana_d{args.dlnprior:.0f}_o{args.order}_r{args.nfreqs}_i" + args.iono  + "_" + args.subset + "_" + args.model + "_"

PSR_DICT = ppa.Load_All_Pulsar_Info()

pulsars = [ ppa.Pulsar( PSR_DICT[psrn] , order = args.order \
                       , iono = args.iono , subset=args.subset \
                        , nfreqs=args.nfreqs ) for psrn in PSR_DICT ]


PSR_NAME_LIST = [psr.PSR_NAME for psr in pulsars]
if args.pulsar >= 0 : 
    pulsars = [pulsars[args.pulsar]]
    tag_ana += pulsars[0].PSR_NAME + '_'
elif args.pulsar == -1:
    pulsars = pulsars

array = ppa.Array(pulsars)
NSS = np.sum( array.NSUBSETS_by_SS )
NPSR = array.NPSR

tag_ana += f"Np{NPSR}" 




#=====================================================#
#    Construct the array                              #
#=====================================================#


if args.mock_method == "white":
    array.DPA = array.Gen_Mock_Data(None,None,'none',seed=args.mock_seed)
    tag = "mock_"+args.mock_method+"_%i"%(args.mock_seed) + tag_ana

elif args.mock_method == "auto":
    array.DPA = array.Gen_Mock_Data(args.mock_lma , args.mock_lSa,"auto",seed=args.mock_seed)
    tag = "mock_"+args.mock_method+"_%.1f_%.1f_%i"%(args.mock_lma,args.mock_lSa,args.mock_seed) + tag_ana

elif args.mock_method == "full":
    array.DPA = array.Gen_Mock_Data(args.mock_lma , args.mock_lSa,"full",seed=args.mock_seed)
    tag = "mock_"+args.mock_method+"_%.1f_%.1f_%i"%(args.mock_lma,args.mock_lSa,args.mock_seed) + tag_ana

elif args.mock_method == "data":
    tag = "data" + tag_ana

else: 
    raise


print(tag)


ones = np.ones(array.NPSR)
zeros = np.zeros(array.NPSR)
if args.model == "n":
    lnlike_sig0_raw = array.Generate_Lnlike_Function( "none" )
    lnlike_sig1_raw = array.Generate_Lnlike_Function( "none" )

elif args.model == "a":
    lnlike_sig0_raw = array.Generate_Lnlike_Function( "auto" )
    lnlike_sig1_raw = array.Generate_Lnlike_Function( "auto" )

elif args.model == "f":
    lnlike_sig0_raw = array.Generate_Lnlike_Function( "full" )
    lnlike_sig1_raw = array.Generate_Lnlike_Function( "full" )

elif args.model == "af":
    lnlike_sig0_raw = array.Generate_Lnlike_Function( "auto" )
    lnlike_sig1_raw = array.Generate_Lnlike_Function( "full" )

elif args.model == "na":
    lnlike_sig0_raw = array.Generate_Lnlike_Function( "none" )
    lnlike_sig1_raw = array.Generate_Lnlike_Function( "auto" )

elif args.model == "nf":
    lnlike_sig0_raw = array.Generate_Lnlike_Function( "none" )
    lnlike_sig1_raw = array.Generate_Lnlike_Function( "full" )
    
else:
    raise
#print(tag)


#=====================================================#
#    Mapping the parameters                           #
#=====================================================#
def Mapper(params):

    l10_EFAC = params[:NSS]
    l10_EQUAD = params[NSS:2*NSS]
    l10_S0 = params[2*NSS:3*NSS]
    Gamma = params[3*NSS:4*NSS]
    sDTE = params[4*NSS : 4*NSS+NPSR]
    l10_ma = params[4*NSS+NPSR ]
    l10_Sa = params[4*NSS+NPSR + 1]

    return l10_EFAC , l10_EQUAD , l10_S0 , Gamma , sDTE  , l10_ma , l10_Sa


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
l10_S0_lp , l10_S0_sp = ppa.gen_uniform_lnprior(-8,2)
Gamma_lp    , Gamma_sp = ppa.gen_uniform_lnprior(-2,0)
sDTE_lp = array.sDTE_LNPRIOR
l10_ma_lp , l10_ma_sp = ppa.gen_uniform_lnprior(args.lma_min,args.lma_max)
l10_Sa_lp , l10_Sa_sp = ppa.gen_uniform_lnprior(-8,2)

def lnprior_nonmodel( params ):
    l10_EFAC , l10_EQUAD , l10_S0 , Gamma,  sDTE , l10_ma , l10_Sa =  Mapper( params )

    LP_1 = np.sum( [l10_EFAC_lp(l10_EFAC[i]) for i in range(NSS)] )
    LP_2 = np.sum( [l10_EQUAD_lp(l10_EQUAD[i]) for i in range(NSS)] )
    LP_3 = np.sum( [l10_S0_lp(l10_S0[i]) for i in range(NSS)] )
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

    #l10_EFAC_bf , l10_EQUAD_bf = array.Load_bestfit_params( )
    sDTE = np.random.rand(len(ones))*0.1 + 1 - 0.05
    init_val = [nmodel_sp()] +[l10_EFAC_sp() for i in range(NSS)]\
                             + [l10_EQUAD_sp() for i in range(NSS)]\
                             + [l10_S0_sp() for i in range(NSS)]\
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
print( lnlike(init))

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
sampler = PTSampler( len(init) ,lnlike,lnprior,groups=groups,cov = cov,resume=True, outDir = name,verbose=True )
sampler.sample(np.array(init),args.nsamp,thin=200)




