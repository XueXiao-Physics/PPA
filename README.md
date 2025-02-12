# First Pulsar Polarization Array Limits on Ultralight Axion-like Dark Matter

This github repository contains the data and code needed to reproduce the main results of the paper:

[First Pulsar Polarization Array Limits on Ultralight Axion-like Dark Matter](https://arxiv.org/abs/2412.02229)
[arXiv:2412.02229](https://arxiv.org/abs/2412.02229)

This code is written in python, which also requires numpy, matplotlib, scipy package and jupyter notebook. 

# Content
|Content   |File   |Description   |
|---|---|---|
| Main script  | ppa/ppa.py  |   |
| Priors for Bayesian analyses  | ppa/priors.py  |   |
| Pulsar properties| ppa/Parfile/..| |
| $\Delta$ PA data | ppa/Data/..  |  |
| RM data from ionFR  | ppa/ionFR_correction/..  |  |
| Directory for MCMC chains |chain_dir.txt| Default location is in the current folder.|
| Run single pulsar noise analyses| run_spa.py| Results will be stored at ppa/Parfile/spa_results.json for future use|
| Run ppa analysis|run_one.py|Bayesian analysis for only one $m_a$|
| Examples|examples/example*.py|Run the full analyses, based on run_one.py|
| Results|examples/Results.ipynb|Pre-derived Results collected from the MCMC chains|

# Citation
@article{xue2024first,
  title={First Pulsar Polarization Array Limits on Ultralight Axion-like Dark Matter},
  author={Xue, Xiao and Dai, Shi and Luu, Hoang Nhan and Liu, Tao and Ren, Jing and Shu, Jing and Zhao, Yue and Zic, Andrew and Bhat, ND and Chen, Zu-Cheng and others},
  journal={arXiv preprint arXiv:2412.02229},
  year={2024}
}

# Result
<img src="https://github.com/XueXiao-Physics/PPA/blob/main/examples/constraint_main.png" width="800"/>

# Imporant Note
1. All angular measurements in the code are expressed in radians, not degrees.
2. Before running the code, it is recommended to set the environment variable for OpenMP to use a single thread. You can do this by executing the following command in your terminal:
   ```
   export OMP_NUM_THREADS=1
   ```
   

