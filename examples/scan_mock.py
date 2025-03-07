import os 
os.chdir("../") 
import numpy as np

# Search properties
det_order = 2
iono = "ionfr" # even we are generating mock data, iono and subset are needed to determine TOA and error bar.
subset = "10cm"
nfreqs = -1
nsamp = 800000

def scanrun(lma_min,lma_max,ipsr=-1):
    argument = "python run_one.py "\
             + f"-if_mock "+if_mock\
             + f" -mock_noise "+ mock_noise\
             + f" -mock_adm " + mock_adm\
             + f" -mock_lma {mock_lma}"\
             + f" -mock_lSa {mock_lSa}"\
             + f" -mock_seed {mock_seed}"\
             + f" -det_order {det_order}"\
             + f" -lma_min {lma_min:.2f} -lma_max {lma_max:.2f}"\
             + f" -dlnprior {dlnprior}" \
             + f" -iono " + iono\
             + f" -subset " + subset\
             + f" -model " + model\
             + f" -nfreqs {nfreqs}"\
             + f" -nsamp {nsamp}"\
             +  " -pulsar " + str(ipsr)   
    if mpi > 1:
            argument = f"mpiexec -np {mpi} "+ argument
        
    os.system(argument + " &")

if_mock = "True"
mpi = 0

mock_noise = "red";mock_adm = "none";mock_lma = -21.5;

mock_lSa = np.log10( 0.3/180 * np.pi )
dlnprior = 0
allseed = np.arange(0,100)


for mock_seed in allseed:
    model = "af" ; scanrun(lma_min=-21.65,lma_max=-21.35)



