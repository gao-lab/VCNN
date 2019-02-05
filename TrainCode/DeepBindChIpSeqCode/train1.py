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


def run_chipseq_data(KernelLen, KernelNum, RandomSeed):
    def get_data_info_list(root_dir = ""):
        #90 in total
        pre = glob.glob(root_dir + "*Gm12878*")
        ret = [it.split("/")[-1].replace("(", "/(") +"/" for it in pre]
        return ret
    cmd = "/home/lijy/anaconda2/bin/ipython ../../corecode/main.py"
    mode_lst = ["vCNN"]

    data_root = "../../Data/ChIPSeqData/HDF5/"
    result_root = "../../OutPutAnalyse/result/ChIPSeq/"
    data_info_lst = get_data_info_list(data_root)
    cell = "chipseq"
    for data_info in data_info_lst:
        for mode in mode_lst:
            data_path = data_root + data_info
            tmp_cmd = str(cmd + " " + data_path + " " + result_root + " " + data_info + " "
                          + mode + " " + KernelLen + " " + KernelNum + " " + RandomSeed + " WorseTest")
            print(tmp_cmd)

            os.system(tmp_cmd)



if __name__ == '__main__':

    # grid search
    ker_size_list = [24]
    number_of_ker_list = [128]
    randomSeedslist = [0, 23, 123, 345, 1234, 9, 2323, 927]
    for RandomSeed in randomSeedslist:
        for KernelNum in number_of_ker_list:
            for KernelLen in ker_size_list:
                run_chipseq_data(str(KernelLen), str(KernelNum), str(RandomSeed))