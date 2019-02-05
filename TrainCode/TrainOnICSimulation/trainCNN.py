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



def run_Simulation_data(KernelLen, KernelNum, RandomSeed):
    def get_data_info_list(root_dir = ""):
        pre = glob.glob(root_dir+"*")

        ret = [it.split("/")[-1].replace("(", "/(") +"/" for it in pre]
        return ret
    cmd = "/home/lijy/anaconda2/bin/ipython ../../corecode/main.py"
    mode_lst = ["CNN"]

    data_root = "../../Data/ICSimulation/HDF5/"
    result_root = "../../OutPutAnalyse/result/ICSimulation/"
    data_info_lst = get_data_info_list(data_root)


    for data_info in data_info_lst:
        for mode in mode_lst:
            data_path = data_root + data_info
            tmp_cmd = str(cmd + " " + data_path + " " + result_root + " " + data_info + " "
                          + mode + " " + KernelLen + " " + KernelNum + " " +RandomSeed)
            print(tmp_cmd)

            os.system(tmp_cmd)


if __name__ == '__main__':

    ker_size_list = [4, 6, 8, 16, 24, 32]
    number_of_ker_list = [80, 100, 120]
    randomSeedslist = [121, 1231, 12341, 1234, 123, 12, 16, 233, 2333, 23, 245, 34561, 3456, 4321, 432, 6]

    for RandomSeed in randomSeedslist:
        for KernelNum in number_of_ker_list:
            for KernelLen in ker_size_list:
                run_Simulation_data(str(KernelLen), str(KernelNum), str(RandomSeed))