import numpy as np
import sys
import os
import time
#lma_idx = float(sys.argv[1])
#dlnprior = float(sys.argv[2])


mock_method = "data"
mock_lma = -22.5
mock_lSa = -1

dlnprior = "\" 0\""
order = "2"
iono = "Subt"
def run(ipsr=None,Range=range(5,56)):
    for lma_idx in Range:
        lma_mid = -24.00 + float(lma_idx)*0.1
        lma_min = lma_mid-0.05
        lma_max = lma_mid+0.05
        argument = "python run_one.py "\
             + "-mock_method "+mock_method\
             + f" -order {order}"\
             + f" -mock_lma {mock_lma:.2f} -mock_lSa {mock_lSa:.2f}"\
             + f" -lma_min {lma_min:.2f} -lma_max {lma_max:.2f}"\
             + f" -dlnprior  " + dlnprior\
             + f" -iono " + iono
        if ipsr ==None:
            pass
        else:
            argument += " -pulsar " + str(ipsr)   
        os.system(argument + " &")

#run(Range=[10])
run()
#for ipsr in range(8,22):
#    run(ipsr)
#    time.sleep(1400)
