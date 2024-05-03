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
TREF = 54550.
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

    def __init__( self, PSR_DICT , order=None , iono=None , nfreqs=5 , subset="all" ):
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
            


        self.TOA = []
        self.DPA = []
        self.DPA_ERR = []
        self.NOBS = []
        self.DES_MTX = []
        self.F_RED = []
        self.FREQS = []
        self.RM = []
        self.RM_ERR = []
        self.WAVE_LENGTH = []

        for SS in self.SUBSETS:
            TOA1 , DPA , DPA_ERR = np.loadtxt( PSR_DICT["DATA"][SS] )
            TOA2,  RM , RM_ERR = np.loadtxt( PSR_DICT["RM"][SS] )
            # Finding overlap TOA
            TOA , idx1 , idx2 = np.intersect1d(TOA1,TOA2, return_indices=True)
            
            DPA = DPA[idx1]
            DPA_ERR = DPA_ERR[idx1]
            RM = RM[idx2]
            RM_ERR = RM_ERR[idx2]

            TOA = TOA - TREF
            NOBS = len(TOA)
            DES_MTX = self.get_design_matrix( TOA , order =  order )
            
            FREQS , F_RED = self.get_F_red( TOA , nfreqs = nfreqs )
            self.F_RED.append(F_RED)
            self.FREQS.append(FREQS)

            
            
            self.TOA.append(TOA)
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


            if iono == "none":
                self.DPA.append(DPA)
                self.DPA_ERR.append(DPA_ERR)
            elif iono == "subt":
                self.DPA.append( DPA + WAVE_LENGTH**2 * RM )
                self.DPA_ERR.append( np.sqrt( DPA_ERR**2 + RM_ERR**2 * WAVE_LENGTH**4 ) )
            else:
                raise



    
    def get_design_matrix( self , TOA , order = 2 ):

        if order in [ 0 , 1 , 2 ]:

            TOA_mid = ( TOA.max() + TOA.min() ) / 2
            TOA_interval = ( TOA.max() - TOA.min() )
            TOA_normalized = ( TOA - TOA_mid ) / TOA_interval

            vec1 = TOA_normalized / TOA_normalized
            vec2 = TOA_normalized
            vec3 = TOA_normalized ** 2

            DES_MTX = np.array( [vec1 , vec2 , vec3] )     
            
            return DES_MTX[:order+1,:]
        
        else:
            print("ISM marginalizasion order: 0, 1 or 2")
            raise
  


    def get_F_red(self , TOA , nfreqs = 0):
        
        if nfreqs > 0 and type(nfreqs)==int :
            F_red = np.zeros( ( nfreqs*2 , len(TOA)) )
            t = TOA * sc.day
            T = ( t.max() - t.min() )
            freqs = 1 / T * np.arange(1,1+nfreqs)
            if nfreqs > 0:
                for i in range( nfreqs ):
                    F_red[2*i,:]   = np.cos( 2 * np.pi * freqs[i] * t )
                    F_red[2*i+1,:] = np.sin( 2 * np.pi * freqs[i] * t )
            return freqs , F_red
        
        else:
            return np.array([]),np.empty((0,len(TOA)))





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
        self.NOBS = [ psr.NOBS for psr in Pulsars]
        self.NOBS_TOTAL = np.sum([ np.sum(NOBS) for NOBS in self.NOBS ])
        

        self.DTE0 = np.array([psr.DTE0 for psr in Pulsars])
        self.sDTE_LNPRIOR = [psr.DTE_LNPRIOR for psr in Pulsars]
        self.PSR_LOC = np.array( [psr.PSR_LOC for psr in Pulsars] )

        self.TOA = [psr.TOA for psr in Pulsars]
        self.DPA = [psr.DPA for psr in Pulsars]
        self.DPA_ERR = [psr.DPA_ERR for psr in Pulsars]
        self.DES_MTX = [ psr.DES_MTX for psr in Pulsars ] 
        self.F_RED = [ psr.F_RED for psr in Pulsars]
        self.FREQS = [ psr.FREQS for psr in Pulsars]
        self.NFREQS = [[ len(x) for x in xs] for xs in self.FREQS  ]



        self.NSUBSETS_by_SS = [ len(SUBSETS) for SUBSETS in self.SUBSETS]
        self.NSUBSETS_TOTAL = np.sum(self.NSUBSETS_by_SS)

        self.F_RED_COMBINED = self.Get_Red_F()
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
    
    # it is precalculated
    def Get_Red_F( self ): 
        ndim_by_psr = [ 2*np.sum(x)  for x in self.NFREQS ] 
        F_RED_COMBINED = []

        for p in range(self.NPSR):
            pointer_x = 0
            pointer_y = 0
            F = np.zeros( ( ndim_by_psr[p] , np.sum(self.NOBS[p]) ) )
            for ss in range(self.NSUBSETS_by_SS[p]):
                dx = self.NFREQS[p][ss]*2
                dy = self.NOBS[p][ss]
                F[pointer_x:pointer_x +  dx,pointer_y:pointer_y + dy] = self.F_RED[p][ss]
                pointer_x += dx
                pointer_y += dy
            F_RED_COMBINED.append(F)

        return  F_RED_COMBINED 



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


    def Get_Phi_inv( self , v , ma, sDTE , Sa2 , Sred2 , adm_signal ):
        # this function is to combine every phi into a combined matrix.
        Sred2 = self.Fold_Array(Sred2)
        Phi_inv_vec = deepcopy( self.PHI_BASE_RED_VEC )
        Phi_logdet = 0
        for p in range(self.NPSR):
            for ss in range(self.NSUBSETS_by_SS[p]):
                Phi_inv_vec[p][ss] = Phi_inv_vec[p][ss] / Sred2[p][ss]
                Phi_logdet += np.sum( np.log( Sred2[p][ss] ) )

        
        if adm_signal == "none":
            Phi_inv = np.diag( np.concatenate([x for xs in Phi_inv_vec for x in xs]) )
            return Phi_inv , Phi_logdet
        
        elif adm_signal in ["auto","full"] :
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

            if adm_signal == "auto":
                for p in range(self.NPSR):
                    sincpq = sincpq_all[p,p];sincp = sincq = sincp_all[p]
                    cpq = cpq_all[p,p]
                    cp = cq = cp_all[p]
                    
                    Phi_ccss = ( 1 + sincpq*cpq - sincp*cp - sincq*cq ) * Sa2
                    Phi_inv_pq = 1 / np.array([ Phi_ccss , Phi_ccss ]) 
                    Phi_inv_vec[p].insert( 0 , Phi_inv_pq )
                    Phi_logdet += np.log(  Phi_ccss )* 2

                Phi_inv = np.diag( np.concatenate([x for xs in Phi_inv_vec for x in xs]) )
                return Phi_inv , Phi_logdet
            
            elif adm_signal == "full":

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
                ADM_Phi_inv_compact , ADM_Phi_logdet = svd_inv(ADM_Phi_compact)
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



    def Get_ADM_F(self,ma ):

        omega = ma*sc.eV/sc.hbar
        F_COMBINED = deepcopy(self.F_RED_COMBINED)
        for p in range(self.NPSR):
            pointer = 0
            F_NEW = np.zeros( (2,len(F_COMBINED[p][0])) )
            for ss in range( self.NSUBSETS_by_SS[p] ):
                NOBS = self.NOBS[p][ss]
                t = self.TOA[p][ss] * sc.day
                F_NEW[ 0 , pointer:pointer+NOBS] = np.cos(omega * t.astype(np.float64))
                F_NEW[ 1 , pointer:pointer+NOBS] = np.sin(omega * t.astype(np.float64))


                pointer += NOBS
            F_COMBINED[p] = np.vstack([ F_NEW , F_COMBINED[p] ])
            
        #return F_matrix  , F_blocks_bysubsets
        return F_COMBINED
    
    #=====================================================#
    #    Load Bestfit parameters                          #
    #=====================================================#

    def Load_bestfit_params(self):
        l10_EFAC = []
        l10_EQUAD = []

        with open(module_path+"Parfile/spa_results.json",'r') as f:
            spa_results = json.load(f)

        for P in range( self.NPSR ):
            PSR_NAME = self.PSR_NAMES[P]
            for S in self.SUBSETS[P]:

                l10_EFAC.append(spa_results[PSR_NAME][S][0])
                l10_EQUAD.append(spa_results[PSR_NAME][S][1])

        return np.array(l10_EFAC) , np.array(l10_EQUAD)
    


    #=====================================================#
    #    Mock Data Module                                 #
    #=====================================================#
    def Gen_White_Mock_Data( self , seed=10):

        np.random.seed(seed)
        l10_EFAC , l10_EQUAD  = self.Load_bestfit_params()
        EFAC = 10 ** l10_EFAC
        EQUAD = 10 ** l10_EQUAD
 

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


    def Gen_Red_Mock_Data( self , method , mock_lma , mock_lSa  , seed=10 ):


        np.random.seed(seed)

        # Generate white noise first
        DPA_white = self.Gen_White_Mock_Data(seed=seed)

        # Get core matrix Phi
        _Phi = self.Get_Phi( 1e-3 , 10**mock_lma ,np.ones(self.NSUBSETS_TOTAL) )
        _Phi_Full = np.block( _Phi.tolist() )  
        _Phi_Auto = np.diag( np.diag( _Phi_Full) )
        F_by_SS = self.Get_F_adm( 10**mock_lma )

        # Gen mock data
        if method == "auto":
            Phi = _Phi_Auto
        elif method == "full":
            Phi = _Phi_Full
        else:
            raise

        _mock = np.random.multivariate_normal(np.zeros(len(Phi)),Phi) * 10**mock_lSa
        mock = _mock@F

        # Divide Data
        pointer = np.concatenate([[0],np.cumsum(np.concatenate(self.NOBS))])    
        iSS = 0
        DPA = np.zeros(self.NPSR,dtype="object")
        for P in range( self.NPSR ):
            DPA_P = []
            for S in range(self.NSUBSETS_by_SS[P]):
                DPA_S = mock[ pointer[iSS] : pointer[iSS+1] ] + DPA_white[P][S]
                DPA_P.append(DPA_S)
                iSS += 1
            DPA[P] = np.array(DPA_P ) 

        
        return DPA
        


    #=====================================================#
    #    For statistics                                   #
    #=====================================================#

    def Generate_Lnlike_Function( self , adm_signal ):    
        #type of signal

        DPA_ERR_by_SS = np.array([ x for xs in self.DPA_ERR for x in xs] ,dtype=np.ndarray)
        DPA_by_SS = [ x for xs in self.DPA for x in xs] 

        DES_MTX_by_SS = [ x for xs in self.DES_MTX for x in xs] 
        ORDERS_by_SS = [ len(x) for x in DES_MTX_by_SS ]

        FREQS_by_SS = [ x for xs in self.FREQS for x in xs]

        NOBS_TOTAL = self.NOBS_TOTAL
        ALL_ORDERS = np.sum(ORDERS_by_SS)
        
        def lnlikelihood( l10_EFAC , l10_EQUAD  ,  l10_S0red , Gamma , sDTE  , l10_ma , l10_Sa ):

            #=============================================#
            #     Mapping Paramaters                      #
            #=============================================#
            EFAC = 10 ** l10_EFAC
            EQUAD = 10 ** l10_EQUAD
            ma = 10 ** l10_ma
            Sa = 10 ** l10_Sa
            S0red = 10 ** l10_S0red

            #=============================================#
            #     White Noise                             #
            #=============================================#
            N_by_SS = ( DPA_ERR_by_SS * EFAC ) ** 2 + EQUAD ** 2
            N_logdet = np.sum( np.log( np.concatenate( N_by_SS ).astype(np.float64) ) )

            #=============================================#
            #     Red Noise                               #
            #=============================================#
            Sred2 = [  np.repeat( (FREQS_by_SS[i]/FYR)**Gamma[i] * S0red[i]**2 *  FREQS_by_SS[i][0]/FYR ,2) for i in range(self.NSUBSETS_TOTAL)]


            Phi_inv , Phi_logdet = self.Get_Phi_inv( 1e-3 , ma , sDTE , Sa**2 , Sred2 , adm_signal )

            #=============================================#
            #     Phi, Axion and Red                      #
            #=============================================#
            # It is very important that Sa and S0 are not too different from each other.
            if adm_signal in ['auto','full']:
                F_by_SS = self.Get_ADM_F( ma )
            elif adm_signal in ['none']:
                F_by_SS = self.F_RED_COMBINED
            else:
                raise

            #=============================================#
            #     For marginalization                     #
            #=============================================#

            xNx = 0
            Fx = []
            FM = []
            FNF = []
            MNM = []
            Mx = []
            iss=0
            for p in range(self.NPSR):
                nss = self.NSUBSETS_by_SS[p]
                NEW_DPA_ERR = np.sqrt(np.concatenate( N_by_SS[iss:iss+nss] ))
                DPA = np.concatenate( DPA_by_SS[iss:iss+nss] )
                M = sl.block_diag( *self.DES_MTX[p] ).T

                # whiten everything
                F_whiten   = F_by_SS[p]/ NEW_DPA_ERR[None,:]
                DPA_whiten = DPA       / NEW_DPA_ERR
                M_whiten   = M         / NEW_DPA_ERR[:,None] 

                # get basic values
                xNx += DPA_whiten@DPA_whiten
                Fx.append( F_whiten @ DPA_whiten )
                FM.append( F_whiten @ M_whiten )
                FNF.append( F_whiten @ F_whiten.T )

                # for M
                MNM.append( M_whiten.T @ M_whiten )
                Mx.append(  M_whiten.T @ DPA_whiten )

                # end
                iss+=nss
            
            FNF = sl.block_diag(*FNF)
            MNM = sl.block_diag(*MNM)
            FM  = sl.block_diag(*FM)
            Mx = np.concatenate(Mx)
            Fx = np.concatenate(Fx)
            #print(FNF.shape,MNM.shape,FM.shape,Mx.shape,Fx.shape)

            #=============================================#
            #     Combine                                 #
            #=============================================#
            

            PhiFNF = Phi_inv + FNF
            PhiFNF_inv,PhiFNF_logdet = svd_inv(PhiFNF)

            lnl_white = -0.5 * xNx  - 0.5*N_logdet  - 0.5 * np.log( 2*np.pi ) * NOBS_TOTAL
            lnl_corr = -0.5 * ( -Fx.T @ PhiFNF_inv @ Fx )  - 0.5 * ( Phi_logdet + PhiFNF_logdet )


            # For marginalization
            
            MCM = MNM - FM.T @ PhiFNF_inv @ FM 
            MCx = Mx -  FM.T @ PhiFNF_inv @ Fx
            MCM_inv , MCM_logdet = svd_inv(MCM)

            lnl_M = 0.5 * MCx.T @ MCM_inv @ MCx - 0.5 * MCM_logdet  + 0.5 * np.log( 2*np.pi ) * ALL_ORDERS

            return lnl_white  + lnl_corr + lnl_M
                
                
                    

        return lnlikelihood

