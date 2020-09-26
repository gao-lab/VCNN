# -*- coding: utf-8 -*-

import os
from multiprocessing import Pool
import sys
import glob
import time
import time

def mkdir(path):
	isExists = os.path.exists(path)
	if not isExists:
		os.makedirs(path)
		return (True)
	else:
		return False


def run_model(data_info, mode):
	cmd = "/home/lijy/anaconda2/bin/ipython "
	
	if mode == "CNN":
		cmd = cmd + " trainCNN.py " + str(data_info)
	elif mode == "vCNN":
		cmd = cmd + " trainVCNN.py " + str(data_info)
	else:
		return
	
	os.system(cmd)

def get_data_info():
	file = open('./WorseKey.txt', 'r')
	data_list = file.readlines()
	return data_list


if __name__ == '__main__':
	
	# GPU_SET = sys.argv[1]
	
	modelType = ["vCNN", "CNN"]
	data_list = get_data_info()

	pool = Pool(processes=20)

	for data_info in data_list:
		for mode in modelType:
			data_info = data_info.replace("\n", "")
			pool.apply_async(run_model, (data_info, mode))
			time.sleep(10)
	pool.close()
	pool.join()

