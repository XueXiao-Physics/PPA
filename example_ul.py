import numpy as np
import sys
import os
import time
#lma_idx = float(sys.argv[1])
#dlnprior = float(sys.argv[2])


mock_method = "data"

dlnprior = "\" 0\""
order = "2"
iono = "subt"#"Subt"
subset = "10cm"
model = "f"
nfreqs = "3"

def run(ipsr=-1,Range=range(0,51)):
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
             +  " -pulsar " + str(ipsr)   
        os.system(argument + " &")

run(Range=[0])
#run()
#for ipsr in range(8,22):
#    run(ipsr=ipsr)
#    time.sleep(1400)
