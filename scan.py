import numpy as np
import sys
import os
lma_idx = float(sys.argv[1])
dlnprior = float(sys.argv[2])

lma_mid = -24.00 + float(lma_idx)*0.1
lma_min = lma_mid-0.05
lma_max = lma_mid+0.05

method = "auto"
mock_lma = -22.5
mock_lSa = -1
os.system("python run_one.py -mock "+method+ f" -mock_lma {mock_lma:.2f} -mock_lSa {mock_lSa:.2f} -lma_min {lma_min:.2f} -lma_max {lma_max:.2f} -dlnprior {dlnprior:.2f}")
#os.system("python run_one.py -mock none -lma_min "+str(lma_min)+" -lma_max "+str(lma_max)+" -dlnprior "+str(dlnprior)+" ")