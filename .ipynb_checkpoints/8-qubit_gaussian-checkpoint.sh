#!/bin/bash
#SBATCH --account=p32574  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --partition=long  ### PARTITION (buyin, short, normal, etc)
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --ntasks-per-node=1 ## how many cpus or processors do you need on each computer
#SBATCH --time=100:00:00 ## how long does this need to run (remember different partitions have restrictions on this param)
#SBATCH --mem-per-cpu=8G ## how much RAM do you need per CPU, also see --mem=<XX>G for RAM per node/computer (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=8-qubit_h2o_clifford_gaussian  ## When you run squeue -u NETID this is how you can identify the job

python --version
python 8-qubit_h2o_clifford_gaussian.py