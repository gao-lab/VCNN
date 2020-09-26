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

def run_model(data_info, KernelLen, KernelNum, RandomSeed):
	cmd = "/home/lijy/anaconda2/bin/ipython ../../corecode/main.py"
	mode = "vCNN"
	cell = "chipseq"

	data_root = "../../Data/ChIPSeqData/HDF5/"
	result_root = "../../OutPutAnalyse/result/ChIPSeq/"
	
	data_path = data_root + data_info + '/'

	####
	output_path = result_root + data_info + "/vCNN/"
	max_ker_len = 40
	modelsave_output_filename = output_path + "/model_KernelNum-" + str(KernelNum) + "_initKernelLen-" + \
	                            str(KernelLen) + "_maxKernelLen-" + str(max_ker_len) + "_seed-" + str(
		RandomSeed) \
	                            + "_batch_size-" + str(100) + ".hdf5"

	tmp_path = modelsave_output_filename.replace("hdf5", "pkl")
	test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")
	if os.path.exists(test_prediction_output):
		return 0, 0

	tmp_cmd = str(cmd + " " + data_path + " " + result_root + " " + data_info + " "
	              + mode + " " + KernelLen + " " + KernelNum + " " + RandomSeed + " chipseq")
	os.system(tmp_cmd)


def get_data_info():
	file = open('./WorseKey.txt', 'r')
	data_list = file.readlines()
	return data_list


if __name__ == '__main__':

	KernelLenlist = [10, 17, 24]
	KernelNumlist = [96, 128]
	data_list = get_data_info()

	if len(sys.argv)>1:

		RandomSeed = int(sys.argv[1])

		# grid search
		for data_info in data_list:
			for KernelLen in KernelLenlist:
				for KernelNum in KernelNumlist:
					data_info =data_info.replace("\n","")
					run_model(data_info, str(KernelLen), str(KernelNum), str(RandomSeed))
	
	else:
		# randomSeedslist = [16, 233, 2333, 23, 245, 34561, 3456, 4321, 12, 567,
		#                    2112, 8748, 7784, 5672, 3696, 8762, 1023, 121, 2030, 6944, 6558,
		#                    8705, 5302, 5777, 4472, 8782, 8669, 6850, 2284, 833, 5070, 3379,
		#                    4268, 1981, 4540, 8236, 7085, 3503, 7289, 1557, 2234, 6987, 3337,
		#                    7171, 7126, 9726, 920, 8957, 6098, 2451]
		randomSeedslist = [0, 23, 123, 345, 1234, 9, 2323, 927, 2112, 8748]


		for data_info in data_list:
			for RandomSeed in randomSeedslist:
				for KernelLen in KernelLenlist:
					for KernelNum in KernelNumlist:
						data_info = data_info.replace("\n", "")
						run_model(data_info, str(KernelLen), str(KernelNum), str(RandomSeed))
