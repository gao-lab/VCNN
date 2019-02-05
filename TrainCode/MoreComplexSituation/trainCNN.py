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
        #160 in total
        pre = glob.glob(root_dir+"*")

        ret = [it.split("/")[-1].replace("(", "/(") +"/" for it in pre]
        return ret
    cmd = "/home/lijy/anaconda2/bin/ipython ../../corecode/main.py"
    mode_lst = ["CNN"]

    data_root = "../../Data/MoreComplexSituation/HDF5/"
    result_root = "../../OutPutAnalyse/result/MoreComplexSituation/"
    data_info_lst = get_data_info_list(data_root)


    for data_info in data_info_lst:
        for mode in mode_lst:
            data_path = data_root + data_info
            tmp_cmd = str(cmd + " " + data_path + " " + result_root + " " + data_info + " "
                          + mode + " " + KernelLen + " " + KernelNum + " " +RandomSeed)
            print(tmp_cmd)

            os.system(tmp_cmd)


if __name__ == '__main__':

    ker_size_list = range(6, 22, 2)
    number_of_ker_list = range(64, 129, 32)
    randomSeedslist = [12, 1234]
    for RandomSeed in randomSeedslist:
        for KernelNum in number_of_ker_list:
            for KernelLen in ker_size_list:
                run_Simulation_data(str(KernelLen), str(KernelNum), str(RandomSeed))