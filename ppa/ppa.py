import numpy as np
import scipy.constants as sc
import numpy.linalg as nl
from .priors import get_psr_loc, gen_DM_dist_lnprior , gen_PX_dist_lnprior, get_DTE, gen_uniform_lnprior
import glob
import scipy.linalg as sl
import json
import os
from copy import deepcopy
module_path = os.path.dirname(__file__) + "/"


KPC_TO_S = sc.parsec*1e3/sc.c
TREFd = 54550.
EARTH_TO_GC= 8.3
GC_LOC = np.array([-0.45545384, -7.24952861, -4.01583076])
FYR = 1/sc.year

def get_lc( v , ma ):
    lc = sc.hbar * sc.c / sc.eV / v / sc.parsec*1e-3 / ma
    return lc

def svd_inv( M ):
    
    u,s,v = nl.svd( M , hermitian=True )
    #print( "%.1e"%(s.max()/s.min()))
    if (s<0).any() :
        return u @ np.diag(1/s) @ v , np.inf
    else:
        return u @ np.diag(1/s) @ v , np.sum(np.log(s))
    



#=========================================================#
#    Load the information of all available pulsars        #
#=========================================================#
def Load_All_Pulsar_Info():

    with open(module_path+"Parfile/pulsars.txt",'r') as f:
        lines = f.readlines()
        newlines = []
        # remove titles
        for line in lines:
            if line[0] != "#":
                newlines.append(line)

        param_dict = {}
        for line in newlines:
            str = line.split()
            psr = str[1]
            RA = str[3]
            DEC = str[6]
            DTE_DM = float(str[9])
            _PX = str[11]
            _PX_ERR = str[12]
            try:
                PX = float(_PX)
                PX_ERR = float(_PX_ERR)
                PX_split = _PX.split('.')
                if len(PX_split)==1:
                    PX_ERR = PX_ERR*1
                elif len(PX_split)==2:
                    digit = np.power( 0.1 , len(PX_split[1]) )
                    PX_ERR = PX_ERR * digit

            except:
                PX = np.nan
                PX_ERR = np.nan

            # Find subsets
            Data_Files = sorted(glob.glob(module_path+"Data/"+psr+"_*.txt"))
            RM_Files = sorted(glob.glob(module_path+"ionFR_correction/"+psr+"_ionFR_*.txt"))
            Band_Names_DATA = [ f.split("/")[-1].split("_")[1].split(".")[0]  for f in Data_Files]
            DATA = { Band_Names_DATA[i]:Data_Files[i] for i in range(len(Data_Files)) }

            Band_Names_RM = [ f.split("/")[-1].split("_")[2].split(".")[0]  for f in RM_Files]
            RM = { Band_Names_RM[i]:RM_Files[i] for i in range(len(RM_Files)) }
            param_dict.update({psr:
                {"PSR":psr,
                "RAJ":RA,
                "DECJ":DEC,
                "DTE_DM":DTE_DM,
                "PX":PX,
                "PX_ERR":PX_ERR,
                "DATA":DATA,
                "RM":RM
                }})

        return param_dict
    

#=========================================================#
#    Pre process the pulsar and put them in a class       #
#=========================================================#
class Pulsar():

    def __init__( self, PSR_DICT , order=None , iono=None , nfreqs_dict={}, white_noise_dict={}, subset="all" ):
        self.PSR_NAME = PSR_DICT["PSR"]
        #self.import_psr()

        self.DTE0,self.DTE_LNPRIOR = get_DTE( PSR_DICT["DTE_DM"] , PSR_DICT["PX"] , PSR_DICT["PX_ERR"] )
        self.PSR_LOC = get_psr_loc( PSR_DICT["RAJ"] , PSR_DICT["DECJ"] )



        #self.SUBSETS = list( PSR_DICT["DATA"].keys() )
        SUBSETS = list( PSR_DICT["DATA"].keys() )
        if subset == "all":
            self.SUBSETS = SUBSETS
        elif type(subset) == str:
            if subset in SUBSETS:
                self.SUBSETS = [subset]
            else:
                self.SUBSETS = []
        elif type(subset) == list:
            self.SUBSETS = []
            for ss in subset:
                if ss in SUBSETS:
                    self.SUBSETS.append(ss)
            


        self.TOAs = []
        self.TOBSs = []
        self.DPA = []
        self.DPA_ERR = []
        self.NOBS = []
        self.DES_MTX = []
        self.F_RED = []
        self.FREQS = []
        self.RM = []
        self.RM_ERR = []
        self.WAVE_LENGTH = []
        self.IONO = []
        self.WN = []


        for SS in self.SUBSETS:
            TOAd1 , DPA , DPA_ERR = np.loadtxt( PSR_DICT["DATA"][SS] )
            TOAd2,  RM , RM_ERR = np.loadtxt( PSR_DICT["RM"][SS] )
            # Finding overlap TOA
            TOAd , idx1 , idx2 = np.intersect1d(TOAd1,TOAd2, return_indices=True)
            
            DPA = DPA[idx1]
            DPA_ERR = DPA_ERR[idx1]
            RM = RM[idx2]
            RM_ERR = RM_ERR[idx2]

            TOAs = ( TOAd - TREFd ) * sc.day
            TOBSs = TOAs.max() - TOAs.min()
            NOBS = len(TOAs)


            if iono+"_"+SS in white_noise_dict.keys():
                WN = white_noise_dict[ iono+"_"+SS ]
            else:
                WN = 1
            
            if iono+"_"+SS in nfreqs_dict.keys():
                NFREQS = nfreqs_dict[ iono+"_"+SS ]
            else:
                NREQS = 0


            FREQS , F_RED = self.get_F_red( TOAs , nfreqs = NFREQS )
            self.F_RED.append(F_RED)
            self.FREQS.append(FREQS)
            self.WN.append(WN)
            DES_MTX = self.get_design_matrix( TOAs , order = order )
            
            self.TOBSs.append(TOBSs)
            self.TOAs.append(TOAs)
            self.NOBS.append(NOBS)
            self.DES_MTX.append(DES_MTX)
            self.RM.append(RM)
            self.RM_ERR.append(RM_ERR)
            if SS == "10cm":
                WAVE_LENGTH = 0.1
                self.WAVE_LENGTH.append(WAVE_LENGTH)
            elif SS == "20cm":
                WAVE_LENGTH = 0.2
                self.WAVE_LENGTH.append(WAVE_LENGTH)
            else:
                print("unknown subset:",SS)
                raise


            if iono == "noiono":
                self.DPA.append(DPA)
                self.DPA_ERR.append(DPA_ERR)
                self.IONO.append("noiono")
            elif iono == "ionfr":
                self.DPA.append( DPA + WAVE_LENGTH**2 * RM )
                self.DPA_ERR.append( np.sqrt( DPA_ERR**2 + RM_ERR**2 * WAVE_LENGTH**4 ) )
                self.IONO.append("ionfr")
            else:
                raise



    
    def get_design_matrix( self , TOAs , order = 2 ):

        if order in [ 0 , 1 , 2 ]:

            TOAs_mid = ( TOAs.max() + TOAs.min() ) / 2
            TOBSs = TOAs.max() - TOAs.min() 
            TOAs_normalized = ( TOAs - TOAs_mid ) / TOBSs

            vec1 = TOAs_normalized / TOAs_normalized
            vec2 = TOAs_normalized
            vec3 = TOAs_normalized ** 2

            DES_MTX = np.array( [vec1 , vec2 , vec3] )     
            
            return DES_MTX[:order+1,:]
        
        else:
            print("ISM marginalizasion order: 0, 1 or 2")
            raise
  


    def get_F_red(self , TOAs , nfreqs = 0):
        
        if nfreqs > 0 and type(nfreqs)==int :
            F_red = np.zeros( ( nfreqs*2 , len(TOAs)) ) 
            TOBSs = TOAs.max() - TOAs.min() 
            freqs = 1 / TOBSs * np.arange(1,1+nfreqs)
            #print(TOBSs , freqs)
            if nfreqs > 0:
                for i in range( nfreqs ):
                    F_red[2*i,:]   = np.cos( 2 * np.pi * freqs[i] * TOAs )
                    F_red[2*i+1,:] = np.sin( 2 * np.pi * freqs[i] * TOAs )
            return freqs , F_red
        
        else:
            return np.array([]),np.empty((0,len(TOAs)))





#=========================================================#
#    Form a list of "Pulsar" objects, make an array       #
#=========================================================#

class Array():

    def __init__( self , Pulsars ):

        # elimate all pulsars that have no data
        Pulsars = [ P for P in Pulsars if len(P.SUBSETS) ]

        # Rearrange subsets
        self.PSR_NAMES = [ psr.PSR_NAME for psr in Pulsars ]
        self.NPSR = len(Pulsars)
        self.SUBSETS = [ psr.SUBSETS for psr in Pulsars ]
        self.IONO = [psr.IONO for psr in Pulsars]
        self.NOBS = [ psr.NOBS for psr in Pulsars]
        self.NOBS_TOTAL = np.sum([ np.sum(NOBS) for NOBS in self.NOBS ])
        self.WN = [psr.WN for psr in Pulsars]
        
        self.DTE0 = np.array([psr.DTE0 for psr in Pulsars])
        self.sDTE_LNPRIOR = [psr.DTE_LNPRIOR for psr in Pulsars]
        self.PSR_LOC = np.array( [psr.PSR_LOC for psr in Pulsars] )

        self.TOAs = [psr.TOAs for psr in Pulsars]
        self.TOBSs = [psr.TOBSs for psr in Pulsars]
        self.DPA = [psr.DPA for psr in Pulsars]
        self.DPA_ERR = [psr.DPA_ERR for psr in Pulsars]
        self.DES_MTX = [ psr.DES_MTX for psr in Pulsars ] 
        self.F_RED = [ psr.F_RED for psr in Pulsars]
        self.FREQS = [ psr.FREQS for psr in Pulsars]
        self.NFREQS = [[ len(x) for x in xs] for xs in self.FREQS  ]

        self.NSUBSETS_by_SS = [ len(SUBSETS) for SUBSETS in self.SUBSETS]
        self.NSUBSETS_TOTAL = np.sum(self.NSUBSETS_by_SS)

        self.PHI_BASE_RED_VEC = self.Get_Phi_base_red()


    # for an array that is by subset, fold it
    def Fold_Array(self,Array_by_ss):
        iss = 0
        New_Array = []
        for p in range(self.NPSR):
            New_Array_psr = []
            for ss in range(self.NSUBSETS_by_SS[p] ):
                New_Array_psr.append( Array_by_ss[iss] )
                iss += 1
            New_Array.append( New_Array_psr )
        return New_Array

        
    #=====================================================#
    #    Get F for red noises, pre-calculated             #
    #=====================================================#
        
    #  It is to put the red noise Phi into a diagonal matrix
    def Get_Phi_base_red( self ):

        PHI_BASE_RED_VEC = []
        for p in range(self.NPSR):
            PHI_BASE_RED_VEC_psr = []
            for ss in range(self.NSUBSETS_by_SS[p]):
                PHI_BASE_RED_VEC_psr.append( np.ones( self.NFREQS[p][ss]*2 ) )
            
            PHI_BASE_RED_VEC.append( PHI_BASE_RED_VEC_psr )

        return PHI_BASE_RED_VEC 



    #=====================================================#
    #    Axion Properties                                 #
    #=====================================================#
    def Get_Star_Locations( self , sDTE ):

        PSR_POS = self.PSR_LOC * sDTE[:,None] * self.DTE0[:,None]
        DL_MTX = np.zeros((self.NPSR,self.NPSR))

        for p in range(self.NPSR):
            for q in range(p+1,self.NPSR):
                l_p = PSR_POS[p]
                l_q = PSR_POS[q]
                DL_MTX[p,q] = DL_MTX[q,p] = np.sqrt(np.power(l_p-l_q,2).sum())


        return PSR_POS,DL_MTX

    #=====================================================#
    #    Phi, for each model                              #
    #=====================================================#

    def Get_Phi_inv_red( self, Sred2 ):
        Sred2 = self.Fold_Array(Sred2)
        Phi_inv_red = deepcopy( self.PHI_BASE_RED_VEC )
        Phi_logdet = 0
        for p in range(self.NPSR):
            for ss in range(self.NSUBSETS_by_SS[p]):
                Phi_inv_red[p][ss] =    Phi_inv_red[p][ss] / Sred2[p][ss]
                Phi_logdet += np.sum( -np.log( Phi_inv_red[p][ss] ) )
        return Phi_inv_red , Phi_logdet
    

    def Get_Phi_ADM(self,v,ma,sDTE,Sa2):

        DTE = self.DTE0 * sDTE
        PSR_POS , DL_MTX = self.Get_Star_Locations( sDTE )
        omega = ma * sc.eV / sc.hbar
        lc = get_lc( v , ma )


        sincpq_all =  np.sinc( DL_MTX / lc / np.pi )
        sincp_all = np.sinc( DTE/ lc / np.pi )
        cp_all = np.cos( omega * DTE * KPC_TO_S )
        sp_all = np.sin( omega * DTE * KPC_TO_S )
        cpq_all = cp_all[:,None]*cp_all[None,:] + sp_all[:,None]*sp_all[None,:]
        spq_all = sp_all[:,None]*cp_all[None,:] - cp_all[:,None]*sp_all[None,:]    
        
        
        ADM_Phi_compact = np.zeros( (self.NPSR*2,self.NPSR*2 ))
        pointer_x = 0
        for p in range(self.NPSR):
            pointer_y = 0
            for q in range(self.NPSR):
                sincpq = sincpq_all[p,q];sincp = sincp_all[p];sincq = sincp_all[q]
                cpq = cpq_all[p,q]; spq = spq_all[p,q]
                cp = cp_all[p] ; sp = sp_all[p] ; cq = cp_all[q] ; sq = sp_all[q]
                
                Phi_ccss = 1 + sincpq*cpq - sincp*cp - sincq*cq
                Phi_sc =  sincpq*spq - sincp*sp + sincq*sq
                Phi_cs = -Phi_sc

                Phi_pq = np.array([[Phi_ccss,Phi_cs],[Phi_sc,Phi_ccss]]) * Sa2
                ADM_Phi_compact[ pointer_x:pointer_x+2 , pointer_y:pointer_y+2 ] = Phi_pq
                pointer_y += 2
            pointer_x += 2
        #ADM_Phi_inv_compact , ADM_Phi_logdet = svd_inv(ADM_Phi_compact)
        # ADM_Phi_inv_compact = sl.inv(ADM_Phi_compact) ; ADM_Phi_logdet = nl.slogdet(ADM_Phi_compact)[1]
        # Phi_logdet += ADM_Phi_logdet

        return ADM_Phi_compact


    def Get_Phi_inv_all( self , v , ma, sDTE , Sa2 , Sred2 , adm_signal  ):
        # this function is to combine every phi into a combined matrix.

        Phi_inv_vec , Phi_logdet = self.Get_Phi_inv_red(Sred2)

        
        if adm_signal in [ "none"]:
            Phi_inv = np.diag( np.concatenate([x for xs in Phi_inv_vec for x in xs]) )
            
        

        elif adm_signal in ["auto","full"]:
            Phi_ADM_compact = self.Get_Phi_ADM( v , ma, sDTE , Sa2 )
            if adm_signal == "auto":
                pointer = 0
                for p in range(self.NPSR):
                    Phi_pq = np.diag( Phi_ADM_compact[pointer:pointer+2 , pointer:pointer+2] )
                    Phi_inv_vec[p].insert( 0 , 1/Phi_pq )
                    Phi_logdet += np.sum(np.log(  Phi_pq ))
                    pointer += 2

                Phi_inv = np.diag( np.concatenate([x for xs in Phi_inv_vec for x in xs]) )
                
            elif adm_signal == "full":
                ADM_Phi_inv_compact = sl.inv(Phi_ADM_compact) ; ADM_Phi_logdet = nl.slogdet(Phi_ADM_compact)[1]
                Phi_logdet += ADM_Phi_logdet

                # now put everything together
                ndim_by_psr = [ 2*np.sum(x)+2 for x in self.NFREQS ]
                Phi_inv = np.zeros( ( np.sum(ndim_by_psr) , np.sum(ndim_by_psr)) )

                pointer_x = 0
                pointer2_x = 0
                for p in range(self.NPSR):
                    pointer_y = 0
                    pointer2_y = 0
                    for q in range(self.NPSR):
                        Phi_inv[ pointer_x:pointer_x+2 , pointer_y:pointer_y+2 ] \
                            = ADM_Phi_inv_compact[ pointer2_x:pointer2_x+2 , pointer2_y:pointer2_y+2]
                        
                        
                        if p == q:
                            phired = np.diag(np.concatenate(Phi_inv_vec[p]))
                            Phi_inv[ pointer_x+2:pointer_x+ndim_by_psr[p] , pointer_y+2:pointer_y+ndim_by_psr[p] ] = phired
                            
                        pointer_y += ndim_by_psr[q] 
                        pointer2_y += 2
                    pointer_x += ndim_by_psr[p] 
                    pointer2_x += 2
        
        return Phi_inv , Phi_logdet

    #=====================================================#
    #    F for axion                                      #
    #=====================================================#

    def Get_F_ADM(self,ma ,adm_signal):

        omega = ma * sc.eV/sc.hbar
        F_ADM = []
        for p in range(self.NPSR):
            F_ADM_psr = []
            for ss in range( self.NSUBSETS_by_SS[p] ):
                NOBS = self.NOBS[p][ss]
                t = self.TOAs[p][ss] 
                if adm_signal in ["auto","full"]:
                    F_ADM_ss = np.zeros((2,NOBS))
                    F_ADM_ss[ 0 ] = np.cos(omega * t.astype(np.float64))
                    F_ADM_ss[ 1 ] = np.sin(omega * t.astype(np.float64))
                    ndim = 2
                elif adm_signal in ["none"]:
                    F_ADM_ss = np.zeros((0,NOBS))
                    ndim = 0

                F_ADM_psr.append(F_ADM_ss)
            F_ADM.append(F_ADM_psr)
            
        #return F_matrix  , F_blocks_bysubsets
        return F_ADM, ndim
    
    #=====================================================#
    #    Load Bestfit parameters                          #
    #=====================================================#

    def Load_bestfit_params(self):
        l10_EFAC = []
        l10_EQUAD = []
        l10_S0red = []
        Gamma = []


        with open(module_path+"Parfile/spa_results.json",'r') as f:
            spa_results = json.load(f)

        for P in range( self.NPSR ):
            psrn = self.PSR_NAMES[P]
            for i,S in enumerate(self.SUBSETS[P]):
                IONO = self.IONO[P][i]
                l10_EFAC.append(spa_results[psrn][IONO+"_"+S][0])
                l10_EQUAD.append(spa_results[psrn][IONO+"_"+S][1])
                l10_S0red.append(spa_results[psrn][IONO+"_"+S][2])
                Gamma.append(spa_results[psrn][IONO+"_"+S][3])

        return np.array(l10_EFAC) , np.array(l10_EQUAD) , np.array(l10_S0red) , np.array(Gamma)
    


    #=====================================================#
    #    Mock Data Module                                 #
    #=====================================================#
    def Gen_White_Mock_Data( self , seed=10):

        np.random.seed(seed)
        l10_EFAC , l10_EQUAD , l10_S0red , Gamma  = self.Load_bestfit_params()
       
        EFAC = 10 ** l10_EFAC
        EQUAD = 10 ** l10_EQUAD
        S0red = 10 ** l10_S0red 

        iSS = 0
        DPA = np.zeros(self.NPSR,dtype="object")
        for P in range( self.NPSR ):
            DPA_P = []
            for S in range(self.NSUBSETS_by_SS[P]):
                DPA_ERR = np.sqrt(self.DPA_ERR[P][S]**2 * EFAC[iSS]**2 + EQUAD[iSS]**2)
                DPA_S = np.random.normal( size = self.NOBS[P][S] ) * DPA_ERR
                iSS += 1
                DPA_P.append(DPA_S)
            DPA[P] =  np.array(DPA_P ) 
        
        return DPA


    def Gen_Mock_Data( self , noise_type="red" , adm_signal="none" , mock_lma = None , mock_lSa = None , seed=10 ):


        np.random.seed(seed)

        # Generate white noise first
        l10_EFAC , l10_EQUAD , l10_S0red , Gamma  = self.Load_bestfit_params()
        DPA = self.Gen_White_Mock_Data(seed=seed)


        S0red = 10**l10_S0red

        # Red, assume nothing
        FREQS_by_SS = [ x for xs in self.FREQS for x in xs]
        TOBSs_by_SS = [x for xs in self.TOBSs for x in xs]
        Sred2 = [  np.repeat( (FREQS_by_SS[i]/FYR)**Gamma[i] * S0red[i]**2 *  1/FYR/TOBSs_by_SS[i] ,2) for i in range(self.NSUBSETS_TOTAL)]
        
        Phi_inv_red = self.Get_Phi_inv_red(Sred2)[0]
        
        F_red = self.F_RED
        

        # here we add noise
        if noise_type == 'white':
            pass
        elif noise_type == "red":
            for p in range(self.NPSR):
                for ss in range(self.NSUBSETS_by_SS[p]):
                    Phi_red_ss = np.diag(1/np.array(Phi_inv_red[p][ss]))
                    if Phi_red_ss.size >0:
                        Fmock = np.random.multivariate_normal(np.zeros(len(Phi_red_ss)),Phi_red_ss)
                        DPA[p][ss] += Fmock @ F_red[p][ss] 
                    else:
                        pass
        else:
            raise

        
        # here we add axion signal

        if adm_signal == "none":
            return DPA
        elif adm_signal in ["full","auto"]:
            ma = 10**mock_lma
            Sa = 10**mock_lSa
            F_adm = self.Get_F_ADM(ma,adm_signal )[0]
            Phi_adm = self.Get_Phi_ADM(1e-3,ma , np.ones(22) , Sa**2)
            if adm_signal == "auto":
                Phi_adm = np.diag(np.diag(Phi_adm))
            elif adm_signal == "full":
                pass
            else:
                raise
            Fmock = np.random.multivariate_normal(np.zeros(len(Phi_adm)),Phi_adm)
            # now devide it
            
            for p in range(self.NPSR):
                for ss in range(self.NSUBSETS_by_SS[p]):
                    DPA[p][ss] += Fmock[2*p : 2*p+2]@F_adm[p][ss]
                p+=2



        return DPA


    #=====================================================#
    #    Finally, likelihood calculation                  #
    #=====================================================#

    def Generate_Lnlike_Function( self , adm_signal ):    
        #type of signal

        DPA_ERR_by_SS = np.array([ x for xs in self.DPA_ERR for x in xs] ,dtype=np.ndarray)
        DPA_by_SS = [ x for xs in self.DPA for x in xs] 

        DES_MTX_by_SS = [ x for xs in self.DES_MTX for x in xs] 
        ORDERS_by_SS = [ len(x) for x in DES_MTX_by_SS ]

        FREQS_by_SS = [x for xs in self.FREQS for x in xs]
        TOBSs_by_SS = [x for xs in self.TOBSs for x in xs]
        WN_by_SS    = np.array([x for xs in self.WN for x in xs])
        print(WN_by_SS)
        NOBS_TOTAL = self.NOBS_TOTAL
        ALL_ORDERS = np.sum(ORDERS_by_SS)
        
        def lnlikelihood( l10_EFAC , l10_EQUAD  ,  l10_S0red , Gamma , sDTE  , l10_ma , l10_Sa ):

            #=============================================#
            #     Mapping Paramaters                      #
            #=============================================#
            EFAC  = 10**np.array(l10_EFAC).astype(float)
            EQUAD = 10**np.array(l10_EQUAD).astype(float)
            ma    = 10**np.array(l10_ma).astype(float)
            Sa    = 10**np.array(l10_Sa).astype(float)
            S0red = 10**np.array(l10_S0red).astype(float)
            
            EFAC[WN_by_SS==0] = 1
            EQUAD[WN_by_SS==0] = 1e-99
        
            #=============================================#
            #     White Noise                             #
            #=============================================#
            N_by_SS = ( DPA_ERR_by_SS * EFAC ) ** 2 + EQUAD ** 2
            N_logdet = np.sum( np.log( np.concatenate( N_by_SS ).astype(np.float64) ) )

            #=============================================#
            #     Red Noise                               #
            #=============================================#

            Sred2 = [  np.repeat( (FREQS_by_SS[i]/FYR)**Gamma[i] * S0red[i]**2 *  1/FYR/TOBSs_by_SS[i] ,2) for i in range(self.NSUBSETS_TOTAL)]                
            Phi_inv , Phi_logdet = self.Get_Phi_inv_all( 1e-3 , ma , sDTE , Sa**2 , Sred2 , adm_signal )
 
            # It is very important that Sa and S0 are not too different from each other.
            F_red_byss = self.F_RED
            F_adm_byss,adm_dim = self.Get_F_ADM(ma,adm_signal)

            #=============================================#
            #     For marginalization                     #
            #=============================================#

            xNx = 0
            Fx = []
            FM = []
            FNF = []
            MNM = []
            MNx = []
            iss = 0
            for p in range(self.NPSR):
                nfreqs_psr = np.sum(self.NFREQS[p])
                FNF_psr = np.zeros( (adm_dim + 2*nfreqs_psr , adm_dim + 2*nfreqs_psr ) )

                FM_adm = []
                FM_red = []

                Fx_adm = 0
                Fx_red = []
                # all these mess are because of block matrices
                for ss in range(self.NSUBSETS_by_SS[p]):

                    nfreqs_ss = self.NFREQS[p][ss]
                    NEW_DPA_ERR = np.sqrt(N_by_SS[iss].astype(np.float64))
                    DPA = DPA_by_SS[iss].astype(np.float64)
                    M = self.DES_MTX[p][ss].astype(np.float64).T
                    

                    F_red_whiten = F_red_byss[p][ss]/ NEW_DPA_ERR[None,:]
                    F_adm_whiten = F_adm_byss[p][ss]/ NEW_DPA_ERR[None,:]

                    DPA_whiten = DPA       / NEW_DPA_ERR
                    M_whiten   = M         / NEW_DPA_ERR[:,None]

                    pointer = adm_dim
                    dpointer = 2 * nfreqs_ss
                    FNF_psr[:adm_dim,:adm_dim] += F_adm_whiten @ F_adm_whiten.T
                    FNF_psr[ pointer : pointer+dpointer , pointer : pointer+dpointer ] = F_red_whiten @ F_red_whiten.T
                    F_ra = F_red_whiten @  F_adm_whiten.T
                    FNF_psr[ pointer : pointer+dpointer , :adm_dim ] = F_ra
                    FNF_psr[:adm_dim , pointer : pointer+dpointer ] = F_ra.T

                    FM_adm.append( F_adm_whiten @ M_whiten )
                    FM_red.append( F_red_whiten @ M_whiten )
                    
                    
                    Fx_adm += F_adm_whiten @ DPA_whiten
                    Fx_red.append( F_red_whiten @ DPA_whiten)

                    MNx.append(  M_whiten.T @ DPA_whiten )
                    MNM.append( M_whiten.T @ M_whiten )
                    
                    xNx += DPA_whiten @ DPA_whiten 
                    iss+=1
                    pointer + dpointer

                FNF.append( FNF_psr )
                FM.append( np.vstack( [np.hstack(FM_adm) , sl.block_diag(*FM_red)] ) )
                Fx.append( np.concatenate(   [ Fx_adm , np.concatenate(Fx_red) ] ) )
                
                
            FNF = sl.block_diag(*FNF)
            MNM = sl.block_diag(*MNM)
            FM  = sl.block_diag(*FM)
            MNx = np.concatenate(MNx)
            Fx = np.concatenate(Fx)

            #=============================================#
            #     Combine                                 #
            #=============================================#
            

            lnl_white = -0.5 * xNx  - 0.5*N_logdet  - 0.5 * np.log( 2*np.pi ) * NOBS_TOTAL
            PhiFNF = Phi_inv + FNF
            MCM = deepcopy(MNM)
            MCx = deepcopy(MNx)

            if Phi_inv.size != 0 :
                
                PhiFNF_inv = sl.inv(PhiFNF) ; PhiFNF_logdet = nl.slogdet(PhiFNF)[1]
                lnl_corr = -0.5 * ( -Fx.T @ PhiFNF_inv @ Fx )  - 0.5 * ( Phi_logdet + PhiFNF_logdet )

                MCM = MCM - FM.T @ PhiFNF_inv @ FM 
                MCx = MCx -  FM.T @ PhiFNF_inv @ Fx
            else:
                lnl_corr = 0

            MCM_inv = sl.inv(MCM) ; MCM_logdet = nl.slogdet(MCM)[1]
            lnl_M = 0.5 * MCx.T @ MCM_inv @ MCx - 0.5 * MCM_logdet  + 0.5 * np.log( 2*np.pi ) * ALL_ORDERS
            lnl = lnl_white  + lnl_corr + lnl_M

            #print(lnl)
            return  lnl
                
                
                    

        return lnlikelihood

