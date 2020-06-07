# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import h5py
import subprocess
import random
import pdb
import os
import glob
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

# idx 模 part_mode == part_num 时候，训练该数据集

def run_Simulation_data(KernelLen, KernelNum, RandomSeed):
    def get_data_info_list(root_dir = "./Demo/*Data"):
        #总共160个
        pre = glob.glob(root_dir+"*")

        ret = [it.split("/")[-1].replace("(", "/(") +"/" for it in pre]
        return ret
    cmd = "python ../../corecode/main.py"
    mode_lst = ["vCNN"]

    data_root = "/rd2/lijy/vCNN/complexSimu/Data/Simu/"
    result_root = "/rd2/lijy/vCNN/complexSimu/result/"
    data_info_lst = get_data_info_list()


    for data_info in data_info_lst:
        for mode in mode_lst:
            data_path = data_root + data_info
            tmp_cmd = str(cmd + " " + data_path + " " + result_root + " " + data_info + " "
                          + mode + " " + KernelLen + " " + KernelNum + " " +RandomSeed)
            print(tmp_cmd)

            os.system(tmp_cmd)


if __name__ == '__main__':

    ker_size_list = range(6, 22, 2)
    number_of_ker_list = range(32, 129, 16)
    randomSeedslist = [12, 1234]
    for RandomSeed in randomSeedslist:
        for KernelNum in number_of_ker_list:
            for KernelLen in ker_size_list:
                run_Simulation_data(str(KernelLen), str(KernelNum), str(RandomSeed))