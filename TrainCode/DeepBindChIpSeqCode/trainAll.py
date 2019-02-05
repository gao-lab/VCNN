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

def run_model(data_info, KernelLen, KernelNum, RandomSeed, GPU_SET):
	cmd =  "/home/lijy/anaconda2/bin/ipython ../../corecode/main.py"
	mode_lst = ["vCNN"]
	cell = "chipseq"


	#data_path = "/rd2/lijy/KDD/vCNNFinal/Data/ChIPSeqData/HDF5/" + data_info + "/"
	data_root = "../../Data/ChIPSeqData/HDF5/"
	result_root = "../../OutPutAnalyse/result/ChIPSeq/"

	for mode in mode_lst:
		data_path = data_root + data_info + '/'
		tmp_cmd = str(cmd + " " + data_path + " " + result_root + " " + data_info + " "
					  + mode + " " + KernelLen + " " + KernelNum + " " + RandomSeed + " " + GPU_SET+ " WorseTest")
		print(tmp_cmd)
		os.system(tmp_cmd)


def get_data_info():
	path = "../../Data/ChIPSeqData/HDF5/"
	path_list = glob.glob(path + '*/')
	data_list = []
	for rec in path_list:
		data_info = rec.split("/")[-2]
		data_list.extend([data_info])
	return data_list


if __name__ == '__main__':

	GPU_SET = sys.argv[1]
	start = int(sys.argv[2])
	end = int(sys.argv[3])

	# grid search
	ker_size_list = [24]
	number_of_ker_list = [128]
	randomSeedslist = [0, 23, 123, 345, 1234, 9, 2323, 927]

	pool = Pool(processes=5)
	data_list = get_data_info()

	print(1)
	for RandomSeed in randomSeedslist:
		for KernelNum in number_of_ker_list:
			for KernelLen in ker_size_list:
				for data_info in data_list[start:end]:
					pool.apply_async(run_model, (data_info, str(KernelLen), str(KernelNum), str(RandomSeed), GPU_SET, ))
	pool.close()
	pool.join()