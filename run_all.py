import numpy as np
import sys
import os
lma_idx = float(sys.argv[1])
dlnprior = float(sys.argv[2])

lma_mid = -24.00 + float(lma_idx)*0.1
lma_min = lma_mid-0.05
lma_max = lma_mid+0.05

os.system("python run_one.py -mock none -lma_min "+str(lma_min)+" -lma_max "+str(lma_max)+" -dlnprior "+str(dlnprior)+" ")
