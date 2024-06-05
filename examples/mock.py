import os 
os.chdir("../") 
import numpy as np
# Mock data properties

# Search properties
order = 2
iono = "ionfr" # even we are generating mock data, iono and subset are needed to determine TOA and error bar.
subset = "10cm"
nfreqs = -1
mpi = 0
nsamp = 5000000

def run(Range=range(0,51),ipsr=-1):
    for lma_idx in Range:
        lma_mid = -23.50 + float(lma_idx)*0.1
        lma_min = lma_mid-0.05
        lma_max = lma_mid+0.05
        argument = "python run_one.py "\
             + f"-if_mock "+if_mock\
             + f" -mock_noise "+ mock_noise\
             + f" -mock_adm " + mock_adm\
             + f" -mock_lma {mock_lma}"\
             + f" -mock_lSa {mock_lSa}"\
             + f" -mock_seed {mock_seed}"\
             + f" -order {order}"\
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

mock_noise = "red";mock_adm = "full";mock_lma = -21.0;
mock_seed = 20




#mock_lSa = -2.00
#dlnprior = 0
#model = "af" ; run()
#model = "nf" ; run()

#dlnprior = 10
#mpi = 0
#mock_seed = 20
#for mock_lSa in np.linspace(-3.0,-2.5,11):
#    model = "nf";run([35])
#    model = "af";run([35])

#mpi = 0
#mock_seed = 20
#mock_lSa  = -2.50
#for dlnprior in np.arange(10,45,2):
#    model="nf";run([42])


mpi = 15
mock_seed = 20
mock_lSa = -2.30
model = "nn" ; dlnprior=0 ; run([25])
model = "ff" ; dlnprior=0 ; run([25])
model = "ff" ; dlnprior=0 ; run([42])
model = "ff" ; dlnprior=0 ; run([48])
