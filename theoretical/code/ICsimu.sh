#!/bin/bash
#SBATCH -J sle223054
#SBATCH -p cn-long
#SBATCH -N 1 
#SBATCH -o ../log/down_%j.out
#SBATCH -e ../log/down_%j.err
#SBATCH --no-requeue
#SBATCH -A gaog_g1
#SBATCH --qos=gaogcnl
#SBATCH -c 1


#/home/gaog_pkuhpc/users/lijy/anaconda3/bin/python ./RunNumericalSimulationOnSimu.py $1
/home/gaog_pkuhpc/users/lijy/anaconda3/bin/python ./RunSimulationICNumerical.py $1 $2
