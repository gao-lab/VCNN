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
def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

# idx 模 part_mode == part_num 时候，训练该数据集

def run_Baaset(RandomSeed, rho=0.99, epsilon=1.0e-8):
    def get_data_info_list(root_dir = ""):
        #160 in total
        pre = glob.glob(root_dir+"*")

        ret = [it.split("/")[-1].replace("(", "/(") +"/" for it in pre]
        return ret
    cmd = "python ../corecode/main.py"
    mode_lst = ["vCNN"]

    data_root = "../../data/encode_roadmap.h5"
    result_root = "../../OutPutAnalyse/result/basset/"



    for mode in mode_lst:
        result_path = result_root
        modelsave_output_prefix = result_path + '/vCNN/'
        modelsave_output_filename = modelsave_output_prefix + "/model_seed-" + str(RandomSeed)+ ".hdf5"
        tmp_path = modelsave_output_filename.replace("hdf5", "pkl")
        test_prediction_output = tmp_path.replace("/model_", "/Report_")
        if os.path.exists(test_prediction_output):
            print("already Trained")
            continue
        tmp_cmd = str(cmd + " " + data_root + " " + result_root + " "
                      + mode+" " +RandomSeed)
        print(tmp_cmd)

        os.system(tmp_cmd)



if __name__ == '__main__':
    import sys


    if len(sys.argv)>1:
        RandomSeed = int(sys.argv[1])
        run_Baaset(str(RandomSeed))
    else:
        randomSeedslist = [121, 1231]
        run_Baaset(str(121))


    