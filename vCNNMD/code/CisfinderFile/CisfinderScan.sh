#!/bin/bash
#SBATCH -J PatternScan
#SBATCH -p cn-long
#SBATCH -N 1
#SBATCH -o ../log/PS_%j.out
#SBATCH -e ../log/PS_%j.err
#SBATCH --no-requeue
#SBATCH -A gaog_g1
#SBATCH --qos=gaogcnl
#SBATCH -c 1


/home/gaog_pkuhpc/users/lijy/Downloads/patternScan -i $1 -f $2 -o $3


