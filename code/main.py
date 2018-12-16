# -*- coding: utf-8 -*-
import os
import h5py
import sys
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[6]
from core_conv_ic import Conv1D_IC
import numpy as np
import unittest
import random
import keras
from build_models import build_vCNN_IC,train_vCNN_IC,train_CNN
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

# tmp_cmd = str(cmd + " " + data_root + " " + result_root + " " + data_info + " " + mode \
#                               + " " + str(seed) + " " + str(0))

data_root = check_dir_end(sys.argv[1])
result_root = check_dir_end(sys.argv[2])
data_info = sys.argv[3]
mode = sys.argv[4]
seed = int(sys.argv[5])
ker_len = 8 # set kernel length as 8

data_dir = check_dir_end(data_root+data_info)
# print(data_dir)
# exit
result_root = check_dir_end(result_root + data_info)
mkdir(result_root)
result_root = check_dir_end(result_root + mode)
mkdir(result_root)

train_dataset = load_dataset(data_dir + "train.hdf5")
test_dataset = load_dataset(data_dir + "test.hdf5")
number_of_ker_list = [128]
batch_size = 100
kernel_max_len = 50
input_shape=test_dataset[0][0].shape
print("input_shape",input_shape)
for filters in number_of_ker_list:
    if mode == "vCNN_IC":
        train_vCNN_IC(input_shape=input_shape, modelsave_output_prefix=result_root,
                      data_set=[train_dataset, test_dataset], number_of_kernel=filters, max_ker_len=kernel_max_len,
                      init_ker_len=ker_len, random_seed=seed, batch_size=batch_size, n_epoch=40, IC_thrd=0.05,
                      jump_trained=True)
    elif mode == "CNN":
        train_CNN(input_shape=input_shape, modelsave_output_prefix=result_root,
                  data_set=[train_dataset, test_dataset], number_of_kernel=filters, kernel_size=ker_len,
                  random_seed=seed, batch_size=batch_size, epoch_scheme=[40])
