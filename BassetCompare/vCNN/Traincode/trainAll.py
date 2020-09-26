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
    
    cmd = "python "
    
    if mode == "basset":
        cmd = cmd + " trainBasset.py " + str(RandomSeed)
    elif mode == "vCNN":
        cmd = cmd + " trainVCNN.py " + str(RandomSeed)
    else:
        return
    
    os.system(cmd)
    





if __name__ == '__main__':

    # GPU_SET = sys.argv[1]
    
    randomSeedslist = [123, 432, 16, 233, 0]
    modelType = ["vCNN","basset"]
    pool = Pool(processes=10)
    
    for RandomSeed in randomSeedslist:
        for mode in modelType:
            # run_model(RandomSeed, mode)
            pool.apply_async(run_model, (RandomSeed,mode))
            time.sleep(40)
    pool.close()
    pool.join()
    
    