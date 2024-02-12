import numpy as np
import scipy.constants as sc
import numpy.linalg as nl
import priors
import glob
import scipy.linalg as sl
#import jax.numpy.linalg as jnl
#import jax
#jax.config.update('jax_platform_name', 'cpu')

KPC_TO_S = sc.parsec*1e3/sc.c
TREF = 54550.
EARTH_TO_GC= 8.3
GC_LOC = np.array([-0.45545384, -7.24952861, -4.01583076])

def get_lc( v , ma ):
    lc = sc.hbar * sc.c / sc.eV / v / sc.parsec*1e-3 / ma
    return lc


#=====================================================#
#    Load the information of all available pulsars    #
#=====================================================#
def Load_Pulsars():

    with open("Parfile/par.txt",'r') as f:
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
            Data_Files = glob.glob("Data/"+psr+"_*.txt")
            Band_Names = [ f.split("/")[1].split("_")[1].split(".")[0]  for f in Data_Files]
            DATA = { Band_Names[i]:Data_Files[i] for i in range(len(Data_Files)) }

            param_dict.update({psr:
                {"PSR":psr,
                "RAJ":RA,
                "DECJ":DEC,
                "DTE_DM":DTE_DM,
                "PX":PX,
                "PX_ERR":PX_ERR,
                "DATA":DATA
                }})

        return param_dict
    

#=====================================================#
#    Pre process the pulsar and put them in a class   #
#=====================================================#
class Pulsar():

    def __init__( self, PSR_DICT  ):
        self.PSR_NAME = PSR_DICT["PSR"]
        #self.import_psr()

        self.DTE,self.DTE_LNPRIOR = priors.get_DTE( PSR_DICT["DTE_DM"] , PSR_DICT["PX"] , PSR_DICT["PX_ERR"] )
        self.PSR_LOC = priors.get_psr_loc( PSR_DICT["RAJ"] , PSR_DICT["DECJ"] )

        self.SUBSETS = list( PSR_DICT["DATA"].keys() )
        self.TOA = []
        self.DPA = []
        self.DPA_ERR = []
        self.NOBS = []
        for SS in self.SUBSETS:
            TOA , DPA , DPA_ERR = np.loadtxt( PSR_DICT["DATA"][SS] )
            NOBS = len(TOA)
            self.TOA.append(TOA)
            self.DPA.append(DPA)
            self.DPA_ERR.append(DPA_ERR)
            self.NOBS.append(NOBS)



    def gen_spa_likelihood( self , SS , p=[-5,5,-8,2]):
        IDX = self.SUBSETS.index(SS)
        DPA = self.DPA[IDX]
        DPA_ERR = self.DPA_ERR[IDX]
        NOBS = self.NOBS[IDX]

        CONST = 0.5*NOBS*np.log(2*np.pi)

        def lnlike( params ):
            l10_EFAC , l10_EQUAD = params
            EFAC = 10**l10_EFAC
            EQUAD = 10**l10_EQUAD
            DPA_ERR_NEW_SQ = ( EFAC * DPA_ERR )**2 + EQUAD**2
            DPA0 = ( DPA / DPA_ERR_NEW_SQ ).sum() / ( 1 / DPA_ERR_NEW_SQ ).sum()
            lnl = - 0.5*np.sum( (DPA - DPA0)**2 / DPA_ERR_NEW_SQ ) - 0.5*np.sum( np.log( DPA_ERR_NEW_SQ ) ) - CONST
            return lnl
        

        l10_EFAC_prior,l10_EFAC_prior_sampler = priors.gen_uniform_lnprior(p[0],p[1])
        l10_EQUAD_prior,l10_EQUAD_sampler = priors.gen_uniform_lnprior(p[2],p[3])
        
        def lnprior( params ):
            l10_EFAC , l10_EQUAD = params
            return l10_EFAC_prior(l10_EFAC)  + l10_EQUAD_prior(l10_EQUAD)
        
        def init_gen():
            init = np.array( [l10_EFAC_prior_sampler() , l10_EQUAD_sampler()] )
            return init

        return lnlike,lnprior,init_gen





#=====================================================#
#    For an list of "Pulsar" project, make an array   #
#=====================================================#

class Array():

    def __init__(self , Pulsars):

        # Rearrange subsets
        self.PSR_NAMES = [ psr.PSR_NAME for psr in Pulsars ]
        self.NPSR = len(Pulsars)
        self.SUBSETS = [ psr.SUBSETS for psr in Pulsars ]
        #self.NSUBSETS = [ len(SUBSETS) for SUBSETS in self.SUBSETS]
        self.NOBS = [ psr.NOBS for psr in Pulsars]
        self.NOBS_TOTAL = np.sum([ np.sum(NOBS) for NOBS in self.NOBS ])
        

        self.DTE = np.array([psr.DTE for psr in Pulsars])
        self.sDTE_LNPRIOR = [psr.DTE_LNPRIOR for psr in Pulsars]
        self.PSR_LOC = np.array( [psr.PSR_LOC for psr in Pulsars] )

        self.TOA = np.array([psr.TOA for psr in Pulsars],dtype=np.ndarray)
        self.DPA = np.array([psr.DPA for psr in Pulsars],dtype=np.ndarray)
        self.DPA_ERR = np.array([psr.DPA_ERR for psr in Pulsars],dtype=np.ndarray)


        self.NSUBSETS_by_SS = [ len(SUBSETS) for SUBSETS in self.SUBSETS]






    def Get_Star_Locations( self , sDTE ):

        PSR_POS = self.PSR_LOC * sDTE[:,None] * self.DTE[:,None]
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
        
        DTE = self.DTE * sDTE
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
                Phi_cs = -sincpq*spq + sincp*sp - sincq*sq

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
                t = self.TOA[P][S]
                Fcos = np.cos(omega * t)
                Fsin = np.sin(omega * t)
                Fss = np.vstack([Fcos,Fsin])
                F_blocks_bysubsets.append(Fss)
                F_matrix[ P*2 : P*2 + 2 , NTOT : NTOT + NOBS] = Fss.copy()

                NTOT += NOBS
            

        return F_matrix  , F_blocks_bysubsets
    

    #=====================================================#
    #    For statistics                                   #
    #=====================================================#

    def Generate_Lnlike_Function( self , method = "Full" ):    
        #type of signal

        
        NSUBSETS = np.sum( self.NSUBSETS_by_SS )

        DPA_ERR_by_SS = np.array( [ x for xs in self.DPA_ERR for x in xs] , dtype="object" )
        DPA_by_SS = np.array( [ x for xs in self.DPA for x in xs] , dtype="object" )
        V_by_SS = np.array( [ x/x for xs in self.DPA for x in xs] , dtype="object" )


        CONST =  0.5 * self.NOBS_TOTAL * np.log( 2*np.pi )

        if method == "Full":
            switch = [1,0]
        elif method == "Auto":
            switch = [0,1]
        
        def lnlikelihood( l10_EFAC , l10_EQUAD , sDTE , vs , l10_ma , l10_Sa ):
            #==========================#
            # Mapping Paramaters       #
            #==========================#
            EFAC = 10 ** l10_EFAC
            EQUAD = 10 ** l10_EQUAD
            ma = 10 ** l10_ma
            Sa = 10 ** l10_Sa

            #==========================#
            # White Noise              #
            #==========================#

            N_by_SS = ( DPA_ERR_by_SS * EFAC ) ** 2 + EQUAD ** 2
            N_logdet = np.sum( np.log( np.concatenate( N_by_SS ) ) )


            #==========================#
            # Axion Signal             #
            #==========================#
            _Phi = self.Get_Sigma( vs*1e-3 , ma , sDTE)
            _Phi_Full = np.block( _Phi.tolist() )  
            _Phi_Auto = np.diag( np.diag(_Phi_Full) )
            Phi = (_Phi_Full*switch[0] + _Phi_Auto*switch[1])*Sa**2

            F,F_by_SS = self.Get_F( ma )
            Phi_sgn , Phi_logdet = nl.slogdet( Phi )

            #==========================#
            # Combine                  #
            #==========================#
            #if ( nl.eigvals(Phi) > 0 ).all() : # Phi has to be positve definite
            try: 
                #============================#
                # For mean value subtraction #
                #============================#
                Fx = np.zeros( ( self.NPSR*2,NSUBSETS ) )
                Fv = np.zeros( ( self.NPSR*2,NSUBSETS ) )
                vNx = np.zeros( NSUBSETS )
                vNv = np.zeros( NSUBSETS )
                xNx = np.zeros( NSUBSETS ) 
                FNF = np.zeros( ( self.NPSR*2 , self.NPSR*2 ) )

                iSS = 0
                for P in range(self.NPSR):
                    FNF_psr = np.zeros((2,2))
                    for i in range(self.NSUBSETS_by_SS[P]):
                        DPA_whiten = DPA_by_SS[iSS] / np.sqrt( N_by_SS[iSS] )
                        V_whiten = V_by_SS[iSS] / np.sqrt( N_by_SS[iSS] )
                        F_whiten = F_by_SS[iSS] / np.sqrt( N_by_SS[iSS] )[None,:] 

                        FNF_psr +=  F_whiten @ F_whiten.T
                        Fx[P*2 : P*2+2 , iSS] = F_whiten @ DPA_whiten
                        Fv[P*2 : P*2+2 , iSS] = F_whiten @ V_whiten
                        vNx[iSS] =  V_whiten @ DPA_whiten
                        vNv[iSS] =  V_whiten @ V_whiten
                        xNx[iSS] =  DPA_whiten @ DPA_whiten
                        iSS += 1
                    FNF[ P*2 : P*2+2 , P*2 : P*2+2 ] = FNF_psr

                Phi_inv = sl.inv( Phi  )
                PhiFNF = Phi_inv + FNF
                PhiFNF_inv = sl.inv( PhiFNF )
                PhiFNF_sgn , PhiFNF_logdet = nl.slogdet( PhiFNF )

                vNx = np.diag(vNx)
                vNv = np.diag(vNv)

                vCx = vNx - Fv.T @ PhiFNF_inv @ Fx
                vCv = vNv - Fv.T @ PhiFNF_inv @ Fv
                vCv_inv = sl.inv(vCv)
                x0_mlh = np.sum( vCv_inv @ vCx ,axis=1)
                #print(x0_mlh)
                #============================#
                # The end of Subtraction     #
                #============================#
                
                Sigma_logdet = Phi_logdet + PhiFNF_logdet + N_logdet
                lnlike_val = -0.5*( xNx.sum() - ( Fx.T @ PhiFNF_inv @ Fx).sum()  - x0_mlh @ vCv @ x0_mlh ) - 0.5*Sigma_logdet - CONST
                #print( x0_mlh)

            except:
                lnlike_val = - np.inf
                print("%.1f"%l10_ma,"inversion fail")
            
            return lnlike_val



        return lnlikelihood




