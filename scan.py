import numpy as np
import sys
import os
#lma_idx = float(sys.argv[1])
#dlnprior = float(sys.argv[2])


mock_method = "data"
mock_lma = -22.5
mock_lSa = -1
dlnprior = "\" -1\""
order = "2"
for lma_idx in [5,6]:#[10,11,12]:#[8,9,10,11,12,13,15]:#range(5,56):
    lma_mid = -24.00 + float(lma_idx)*0.1
    lma_min = lma_mid-0.05
    lma_max = lma_mid+0.05
    os.system("python run_one.py -mock_method "+mock_method+ f" -order {order}"+\
            f" -mock_lma {mock_lma:.2f} -mock_lSa {mock_lSa:.2f}\
            -lma_min {lma_min:.2f} -lma_max {lma_max:.2f}\
            -dlnprior  " + dlnprior +" &")
