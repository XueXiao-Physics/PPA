import numpy as np
import sys
import os
import time


if_mock = "False"

dlnprior = 0
order = 0
iono = "noiono"
subset = "10cm"
model = "ff"
nfreqs = 0
mpi = 0
nsamp = 500000

def run( Range=range(0,51),ipsr=-1):
    for lma_idx in Range:
        lma_mid = -23.50 + float(lma_idx)*0.1
        lma_min = lma_mid-0.05
        lma_max = lma_mid+0.05
        argument = "python run_one.py "\
             + f"-if_mock "+if_mock\
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

for ipsr in [25,26]:
#for ipsr in [0,5,14,21,13,17,25]:
#for ipsr in [0,5,14,21]:
#for ipsr in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21, 22, 25, 26]:
    run(ipsr=ipsr)
    time.sleep(500)
