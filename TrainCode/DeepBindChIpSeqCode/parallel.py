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


def run_model(RandomSeed, mode):
	cmd = "/home/lijy/anaconda2/bin/ipython "
	
	if mode == "CNN":
		cmd = cmd + " trainCNN.py " + str(RandomSeed)
	elif mode == "vCNN":
		cmd = cmd + " trainVCNN.py " + str(RandomSeed)
	else:
		return
	
	os.system(cmd)


if __name__ == '__main__':
	
	# GPU_SET = sys.argv[1]
	
	randomSeedslist = [0, 23, 123, 345, 1234, 9, 2323, 927]
	modelType = ["vCNN"]
	pool = Pool(processes=8)
	for RandomSeed in randomSeedslist:
		for mode in modelType:
			pool.apply_async(run_model, (RandomSeed, mode))
			time.sleep(5)
	pool.close()
	pool.join()




