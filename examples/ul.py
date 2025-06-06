import os
os.chdir("../")


if_mock = "False"

dlnprior = 0
det_order = 2
subset = "10cm"
nfreqs = -1
mpi = 0
nsamp = 5000000

def run( Range=range(0,51),ipsr=-1):
    for lma_idx in Range:
        lma_mid = -23.50 + float(lma_idx)*0.1
        lma_min = lma_mid-0.05
        lma_max = lma_mid+0.05
        argument = "python run_one.py "\
             + f" -if_mock "+if_mock\
             + f" -det_order {det_order}"\
             + f" -lma_min {lma_min:.2f} -lma_max {lma_max:.2f}"\
             + f" -dlnprior {dlnprior}"\
             + f" -iono " + iono\
             + f" -subset " + subset\
             + f" -model " + model\
             + f" -nfreqs {nfreqs}"\
             + f" -nsamp {nsamp}"\
             +  " -pulsar " + str(ipsr)   
        if mpi != 0:
            argument = f"mpiexec -np {mpi} "+ argument
        os.system(argument + " &")
 
iono = "ionfr"
model = "aa"
run(Range=range(0,21))
model = "ff"
run(Range=range(0,21))

#run(range(0,51),-1)
