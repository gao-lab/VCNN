# -*- coding: utf-8 -*-

import os
from multiprocessing import Pool
import sys
import glob
import time


def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

def run_model(RandomSeed,mode):
    
    cmd = "/home/lijy/anaconda2/bin/ipython "
    
    if mode == "CNN":
        cmd = cmd + " trainCNN.py " + str(RandomSeed)
    elif mode == "vCNN":
        cmd = cmd + " trainVCNN.py " + str(RandomSeed)
    else:
        return
    print(cmd)
    os.system(cmd)
    





if __name__ == '__main__':

    # GPU_SET = sys.argv[1]
    
    randomSeedslist = [121, 1231, 12341, 1234, 123, 432, 16, 233, 2333, 23, 245, 34561, 3456, 4321,12, 567]
    modelType = ["CNN", "vCNN"]
    pool = Pool(processes=32)
    
    for RandomSeed in randomSeedslist:
        for mode in modelType:
            # run_model(RandomSeed, mode)
            pool.apply_async(run_model, (RandomSeed,mode))
    pool.close()
    pool.join()
    
    