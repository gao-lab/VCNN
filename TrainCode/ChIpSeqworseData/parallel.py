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
	
	# randomSeedslist = [16, 233, 2333,  245, 34561, 3456, 4321, 12, 567,
	#                    2112, 8748, 7784, 5672, 3696, 8762, 1023,  121, 2030, 6944, 6558,
	# 			       8705, 5302, 5777, 4472, 8782, 8669, 6850, 2284,  833, 5070, 3379,
	# 			       4268, 1981, 4540, 8236, 7085, 3503, 7289, 1557, 2234, 6987, 3337,
	# 			       7171, 7126, 9726,  920, 8957, 6098, 2451,23,
	#                    ]
	# modelType = ["vCNN", "CNN"]
	randomSeedslist = [0, 23, 123, 345, 1234, 9, 2323, 927]

	modelType = ["vCNN"]
	for i in range(1):
		randomlist = randomSeedslist[i*10:(i+1)*10]
		pool = Pool(processes=10)

		for RandomSeed in randomlist:
			for mode in modelType:
				pool.apply_async(run_model, (RandomSeed, mode))
				time.sleep(5)
		pool.close()
		pool.join()

