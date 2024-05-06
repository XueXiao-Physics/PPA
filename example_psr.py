import numpy as np
import sys
import os
import time


mock_method = "data"

dlnprior = "\" 0\""
order = "2"
iono = "none"
subset = "10cm"
model = "f"
nfreqs = "3"
mpi = 0
nsamp = 500000

def run( Range=range(0,51),ipsr=-1):
    for lma_idx in Range:
        lma_mid = -23.50 + float(lma_idx)*0.1
        lma_min = lma_mid-0.05
        lma_max = lma_mid+0.05
        argument = "python run_one.py "\
             + "-mock_method "+mock_method\
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
        
for ipsr in range(0,30):
    try:
        run(ipsr=ipsr)
        time.sleep(350)
    except:
        pass
