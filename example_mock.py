import numpy as np
import sys
import os
import time
#lma_idx = float(sys.argv[1])
#dlnprior = float(sys.argv[2])


# Mock data properties
mock_method = "auto"
mock_lma = "-22.0"
mock_lSa = "-3.0"
mock_seed = "22"


# Search properties
dlnprior = "\" 0\""
order = "2"
iono = "subt"#"Subt"
subset = "10cm"
model = "nf"
nfreqs = "3"
mpi = 0
nsamp = 5000000

def run(Range=range(0,51),ipsr=-1):
    for lma_idx in Range:
        lma_mid = -23.50 + float(lma_idx)*0.1
        lma_min = lma_mid-0.05
        lma_max = lma_mid+0.05
        argument = "python run_one.py "\
             + "-mock_method "+mock_method\
             + f" -mock_lma " + mock_lma\
             + f" -mock_lSa " + mock_lSa\
             + f" -mock_seed " + mock_seed\
             + f" -order {order}"\
             + f" -lma_min {lma_min:.2f} -lma_max {lma_max:.2f}"\
             + f" -dlnprior " + dlnprior\
             + f" -iono " + iono\
             + f" -subset " + subset\
             + f" -model " + model\
             + f" -nfreqs " + nfreqs\
             + f" -nsamp {nsamp}"\
             +  " -pulsar " + str(ipsr)   
        if mpi != 0:
            argument = f"mpiexec -np {mpi} "+ argument
        
        os.system(argument + " &")

run([3,  4,  5,  6,  7, 16, 42, 45, 48],-1)
#run(Range=np.arange(0,30))
#run()
#for ipsr in range(8,22):
#    run(ipsr=ipsr)
#    time.sleep(1400)