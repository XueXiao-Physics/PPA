import numpy as np
import scipy.constants as sc
import numpy.linalg as nl
from .priors import get_psr_loc, gen_DM_dist_lnprior , gen_PX_dist_lnprior, get_DTE, gen_uniform_lnprior
import glob
import scipy.linalg as sl
import json
import os
module_path = os.path.dirname(__file__) + "/"


KPC_TO_S = sc.parsec*1e3/sc.c
TREF = 54550.
EARTH_TO_GC= 8.3
GC_LOC = np.array([-0.45545384, -7.24952861, -4.01583076])

def get_lc( v , ma ):
    lc = sc.hbar * sc.c / sc.eV / v / sc.parsec*1e-3 / ma
    return lc

def svd_inv( M ):
    
    u,s,v = nl.svd( M , hermitian=True )
    if (s<0).any() :#or s.max()/s.min()>1e15 :
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

    def __init__( self, PSR_DICT , order=None , iono=None , subset="all" ):
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

            NOBS = len(TOA)
            if order in [ 0 , 1 , 2 ]:
                DES_MTX = self.get_design_matrix( TOA , order =  order )
            else:
                print("ISM marginalizasion order: 0, 1 or 2")
                raise


            self.TOA.append(TOA-TREF)
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

        TOA_mid = ( TOA.max() + TOA.min() ) / 2
        TOA_interval = ( TOA.max() - TOA.min() )
        TOA_normalized = ( TOA - TOA_mid ) / TOA_interval

        vec1 = TOA_normalized / TOA_normalized
        vec2 = TOA_normalized
        vec3 = TOA_normalized ** 2

        DES_MTX = np.array( [vec1 , vec2 , vec3] )     
        
        return DES_MTX[:order+1,:]          




    def gen_spa_likelihood( self , SS , p=[-2,2,-8,2 ] ):

        SS_IDX = self.SUBSETS.index(SS)
        DPA = self.DPA[SS_IDX]
        DPA_ERR = self.DPA_ERR[SS_IDX]
        NOBS = self.NOBS[SS_IDX]
        DES_MTX = self.DES_MTX[SS_IDX]
        def lnlike( params ):
            l10_EFAC , l10_EQUAD  = params
            EFAC = 10**l10_EFAC
            EQUAD = 10**l10_EQUAD
            DPA_ERR_NEW_SQ = ( EFAC * DPA_ERR )**2 + EQUAD**2
            DPA_ERR_NEW = np.sqrt( DPA_ERR_NEW_SQ )

            # original likelihood
            DPA_whiten = DPA / DPA_ERR_NEW
            lnl_original = - 0.5 * np.sum( DPA_whiten**2  ) - 0.5 * np.sum( np.log( DPA_ERR_NEW_SQ ) ) -  0.5  * np.log(2*np.pi) * NOBS

            # after marginalization

            DES_MTX_whiten = DES_MTX / DPA_ERR_NEW[None,:]
            MCM =  DES_MTX_whiten @ DES_MTX_whiten.T
            MCM_inv , MCM_logdet = svd_inv( MCM )

            DPA_DES = DPA_whiten @ DES_MTX_whiten.T
            lnl_marginalized = 0.5 * DPA_DES @ MCM_inv @ DPA_DES.T - 0.5 * MCM_logdet +  0.5  * np.log(2*np.pi) * len(DES_MTX)


            return lnl_original + lnl_marginalized
            
        l10_EFAC_prior,l10_EFAC_prior_sampler = gen_uniform_lnprior(p[0],p[1])
        l10_EQUAD_prior,l10_EQUAD_sampler = gen_uniform_lnprior(p[2],p[3])
        
        def lnprior( params ):
            l10_EFAC , l10_EQUAD  = params
            return l10_EFAC_prior(l10_EFAC)  + l10_EQUAD_prior(l10_EQUAD) 
        
        def init_gen():
            init = np.array( [l10_EFAC_prior_sampler() , l10_EQUAD_sampler()] )
            return init

        return lnlike,lnprior,init_gen





#=========================================================#
#    For an list of "Pulsar" objects, make an array       #
#=========================================================#

class Array():

    def __init__( self , Pulsars  ):

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
        #self.DES_MTX = np.array( [ psr.DES_MTX for psr in Pulsars ] , dtype=np.ndarray )
        self.DES_MTX = [ psr.DES_MTX for psr in Pulsars ] 


        self.NSUBSETS_by_SS = [ len(SUBSETS) for SUBSETS in self.SUBSETS]
        self.NSUBSETS_TOTAL = np.sum(self.NSUBSETS_by_SS)






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
    #    Axion Properties                                 #
    #=====================================================#


    def Get_Sigma( self , v , ma, sDTE ):
        
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

        Phi_blocks = np.zeros((self.NPSR,self.NPSR),dtype=np.ndarray)

        for p in range(self.NPSR):
            for q in range(self.NPSR):

                sincpq = sincpq_all[p,q]
                sincp = sincp_all[p]
                sincq = sincp_all[q]

                cpq = cpq_all[p,q]
                spq = spq_all[p,q]
                cp = cp_all[p]
                sp = sp_all[p]
                cq = cp_all[q]
                sq = sp_all[q]
                
                Phi_ccss = 1 + sincpq*cpq - sincp*cp - sincq*cq
                Phi_sc =  sincpq*spq - sincp*sp + sincq*sq
                Phi_cs = -Phi_sc

                Phi_pq = np.array([[Phi_ccss,Phi_cs],[Phi_sc,Phi_ccss]]) 

                Phi_blocks[p,q] = Phi_pq.copy()

        return Phi_blocks

    def Get_F(self,ma):

        F_matrix = np.zeros(shape = (self.NPSR*2,self.NOBS_TOTAL))
        F_blocks_bysubsets = []
        omega = ma*sc.eV/sc.hbar
        NTOT = 0

        for P in range(self.NPSR):
            for S in range( self.NSUBSETS_by_SS[P] ):

                NOBS = self.NOBS[P][S]
                t = self.TOA[P][S] * sc.day
                Fcos = np.cos(omega * t.astype(np.float64))
                Fsin = np.sin(omega * t.astype(np.float64))
                Fss = np.vstack([Fcos,Fsin])
                F_blocks_bysubsets.append(Fss)
                F_matrix[ P*2 : P*2 + 2 , NTOT : NTOT + NOBS] = Fss.copy()

                NTOT += NOBS
            

        return F_matrix  , F_blocks_bysubsets
    
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
        _Phi = self.Get_Sigma( 1e-3 , 10**mock_lma ,np.ones(self.NSUBSETS_TOTAL) )
        _Phi_Full = np.block( _Phi.tolist() )  
        _Phi_Auto = np.diag( np.diag( _Phi_Full) )
        F,F_by_SS = self.Get_F( 10**mock_lma )

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

    def Generate_Lnlike_Function( self , method = "Full" ):    
        #type of signal

        DPA_ERR_by_SS = np.array([ x for xs in self.DPA_ERR for x in xs] ,dtype=np.ndarray)
        DPA_by_SS = [ x for xs in self.DPA for x in xs] 
        DES_MTX_by_SS = [ x for xs in self.DES_MTX for x in xs] 
        ORDERS_by_SS = [ len(x) for x in DES_MTX_by_SS ]

        NOBS_TOTAL = self.NOBS_TOTAL
        ALL_ORDERS = np.sum(ORDERS_by_SS)
        
        def lnlikelihood( l10_EFAC , l10_EQUAD  , sDTE  , l10_ma , l10_Sa ):

            #=============================================#
            #     Mapping Paramaters                      #
            #=============================================#
            EFAC = 10 ** l10_EFAC
            EQUAD = 10 ** l10_EQUAD
            ma = 10 ** l10_ma
            Sa = 10 ** l10_Sa

            #=============================================#
            #     White Noise                             #
            #=============================================#

            N_by_SS = ( DPA_ERR_by_SS * EFAC ) ** 2 + EQUAD ** 2
            N_logdet = np.sum( np.log( np.concatenate( N_by_SS ).astype(np.float64) ) )


            #=============================================#
            #     Axion Signal                            #
            #=============================================#
            _Phi = self.Get_Sigma( 1e-3 , ma , sDTE)
            _Phi_Full = np.block( _Phi.tolist() )  
            _Phi_Auto = np.diag( np.diag(_Phi_Full) )

            if method == "Full":
                Phi = _Phi_Full * Sa**2
            elif method == "Auto":
                Phi = _Phi_Auto * Sa**2
            elif method == "None":
                Phi = _Phi_Auto * 1e-99
            else:
                raise

            F,F_by_SS = self.Get_F( ma )
            


            #=============================================#
            #     For marginalization                     #
            #=============================================#


            iSS = 0
            iPAR = 0

            FNF = np.zeros( ( self.NPSR*2 , self.NPSR*2 ) )
            Fx  = np.zeros( ( self.NPSR*2  ) )
            FM  = np.zeros( ( self.NPSR*2 , ALL_ORDERS ) )
            MNM = np.zeros( ( ALL_ORDERS , ALL_ORDERS ) )
            Mx  = np.zeros( ( ALL_ORDERS ) )
            xNx = 0

            for P in range(self.NPSR):
                FNF_psr = np.zeros((2,2))
                Fx_psr = np.zeros(2)
                for i in range(self.NSUBSETS_by_SS[P]):

                    ORDER_SS = ORDERS_by_SS[iSS]
                    DPA_SS = DPA_by_SS[iSS].astype(np.float64) 
                    F_SS = F_by_SS[iSS].astype(np.float64)
                    sqrt_N_SS = np.sqrt( N_by_SS[iSS].astype(np.float64) )
                    M_SS = DES_MTX_by_SS[iSS].astype(np.float64)
                    
 
                    DPA_whiten     = DPA_SS     / sqrt_N_SS
                    F_whiten       = F_SS       / sqrt_N_SS[None,:]
                    M_whiten       = M_SS       / sqrt_N_SS[None,:]

                    FM[P*2 : P*2+2 , iPAR : iPAR+ORDER_SS ] = F_whiten @ M_whiten.T
                    MNM[iPAR : iPAR+ORDER_SS,iPAR : iPAR+ORDER_SS] = M_whiten @ M_whiten.T
                    Mx[iPAR : iPAR+ORDER_SS] = M_whiten @ DPA_whiten

                    xNx     +=  DPA_whiten @ DPA_whiten
                    FNF_psr +=  F_whiten @ F_whiten.T
                    Fx_psr  +=  F_whiten @ DPA_whiten

                    iSS += 1
                    iPAR += ORDER_SS

                FNF[ P*2 : P*2+2 , P*2 : P*2+2 ] = FNF_psr
                Fx[  P*2 : P*2+2 ]               = Fx_psr


            Phi_inv,Phi_logdet = svd_inv(Phi)
            PhiFNF = Phi_inv + FNF
            PhiFNF_inv,PhiFNF_logdet = svd_inv(PhiFNF)
            
            xCx = xNx - Fx.T @ PhiFNF_inv @ Fx
            MCM = MNM - FM.T @PhiFNF_inv @ FM
            MCx = Mx -  FM.T @ PhiFNF_inv @ Fx
            MCM_inv , MCM_logdet = svd_inv(MCM)
            

            All_logdet = Phi_logdet + PhiFNF_logdet + N_logdet

            lnl_original = -0.5 * xCx - 0.5 * All_logdet - 0.5 * np.log( 2*np.pi ) * NOBS_TOTAL
            
            lnl_marginalized = 0.5 * MCx.T @ MCM_inv @ MCx - 0.5 * MCM_logdet  + 0.5 * np.log( 2*np.pi ) * ALL_ORDERS



            return lnl_original + lnl_marginalized



        return lnlikelihood

