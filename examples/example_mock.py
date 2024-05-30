import os
os.chdir("../")
import numpy as np
import sys
import time
#lma_idx = float(sys.argv[1])
#dlnprior = float(sys.argv[2])


# Mock data properties

# Search properties
dlnprior = 0
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
        if mpi != 0:
            argument = f"mpiexec -np {mpi} "+ argument
        
        os.system(argument + " &")


if_mock = "True"
mpi = 0
mock_noise = "red";mock_adm = "full";mock_lma = -22.0;mock_seed=20
for i in range(1):
    mock_lSa = -2.80  ; dlnprior = 0
    model = "nf";run()
    model = "af";run()



#mock_noise = "red";mock_adm = "full";mock_lma = -23.0;mock_seed=20
#for i in range(2):
#    mock_lSa = -2.50 + i*0.05 ; dlnprior = 7
#    model = "nf";run([0,1,2,3,4,5,6])
#    model = "af";run([0,1,2,3,4,5,6])

