#!/bin/bash
#SBATCH -J vCNN
#SBATCH -p cn-long
#SBATCH -N 1
#SBATCH -o ../log/vCNN_%j.out
#SBATCH -e ../log/vCNN_%j.err
#SBATCH --no-requeue
#SBATCH -A gaog_g1
#SBATCH --qos=gaogcnl
#SBATCH -c 1


/home/gaog_pkuhpc/users/lijy/anaconda3/bin/python mergePeaks.py $1
