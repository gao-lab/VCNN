# -*- coding: utf-8 -*-
'''
run on simulation data set
using a bash file to pass parameters:
    data_root: the dir where data is saved
    result_root: the dir where result is save
    data_info: the data set's name
    mode: either "CNN" or "vCNN_IC"
    seed: random seed
    gpu_usage: which gpu to use, "0" for default
'''
import os
import h5py
import sys
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[6]
from core_conv_ic import Conv1D_IC
import numpy as np
import unittest
import random
import keras
from build_models import train_vCNN_IC,train_CNN
from my_history import Histories
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import roc_auc_score
import pickle

def mkdir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return(True)
    else :
        return(False)
# make sure the dir is end with "/"
def check_dir_end(dir):
    dir = dir.replace("//","/")
    if not dir[-1] == "/":
        return dir +"/"
    else:
        return dir
# make the path str valid
def modify_path_str(str):
    return str.replace("\n","").replace("//","/")

def scheduler(epoch):
	if epoch <= 20:
		return 0.01
	elif epoch%20==0:
		return 0.1/(epoch*2.0)
	else:
		return 0.1/((epoch-epoch%10+10)*2.0)

def load_dataset(dataset):
    data = h5py.File(dataset, 'r')
    sequence_code = data['sequences'].value
    label = data['labs'].value
    return ([sequence_code, label])

# organization of sys.argv
# data_root + " " + result_root + " " + data_info + " " + mode \
# + " " + seed + " " + gpu_usage

data_root = check_dir_end(sys.argv[1])
result_root = check_dir_end(sys.argv[2])
data_info = sys.argv[3]
mode = sys.argv[4]
seed = int(sys.argv[5])

data_dir = check_dir_end(data_root+data_info)

result_root = check_dir_end(result_root + data_info)
mkdir(result_root)
result_root = check_dir_end(result_root + mode)
mkdir(result_root)

train_dataset = load_dataset(data_dir + "training_set.hdf5")
test_dataset = load_dataset(data_dir + "test_set.hdf5")
number_of_ker_list = [100]
ker_len_lst = [8,16,24,32]
batch_size = 100
kernel_max_len = 50
input_shape=test_dataset[0][0].shape
print("input_shape",input_shape)
for filters in number_of_ker_list:
    for ker_len in ker_len_lst:
        if mode == "vCNN_IC":
            train_vCNN_IC(input_shape=input_shape, modelsave_output_prefix=result_root,
                          data_set=[train_dataset, test_dataset], number_of_kernel=filters, max_ker_len=kernel_max_len,
                          init_ker_len=ker_len, random_seed=seed, batch_size=batch_size, n_epoch=40, IC_thrd=0.05,
                          jump_trained=True)
        elif mode == "CNN":
            train_CNN(input_shape=input_shape, modelsave_output_prefix=result_root,
                          data_set=[train_dataset, test_dataset], number_of_kernel=filters, kernel_size = ker_len,
                 random_seed=seed, batch_size=batch_size, epoch_scheme=[40])
