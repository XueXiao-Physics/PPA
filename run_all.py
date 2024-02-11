import numpy as np
import sys
import os
lma_idx = float(sys.argv[1])
dlnlike = float(sys.argv[2])
log10ma_mid = -24.00 + float(lma_idx)*0.1
l10ma_min = log10ma_mid-0.05
l10ma_max = log10ma_mid+0.05

os.system("python run_one.py "+str(l10ma_min)+" "+str(l10ma_max)+" "+str(dlnlike)+" ")
