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
	result_root = "../../OutPutAnalyse/result/ChIPSeq500/"
	
	data_path = data_root + data_info + '/'
	
	tmp_cmd = str(cmd + " " + data_path + " " + result_root + " " + data_info + " "
	              + mode + " " + KernelLen + " " + KernelNum + " " + RandomSeed + " chipseq")
	os.system(tmp_cmd)


def get_data_info():
	file = open('./WorseKey.txt', 'r')
	data_list = file.readlines()
	return data_list


if __name__ == '__main__':

	data_info = sys.argv[1]
	randomSeedslist = [16, 233, 2333, 23, 245, 34561, 3456, 4321, 12, 567,
	                   2112, 8748, 7784, 5672, 3696, 8762, 1023,  121, 2030, 6944, 6558,
				       8705, 5302, 5777, 4472, 8782, 8669, 6850, 2284,  833, 5070, 3379,
				       4268, 1981, 4540, 8236, 7085, 3503, 7289, 1557, 2234, 6987, 3337,
				       7171, 7126, 9726,  920, 8957, 6098, 2451]

	# grid search
	KernelLen = 24
	KernelNum = 128
	

	for RandomSeed in randomSeedslist:
		run_model(data_info, str(KernelLen), str(KernelNum), str(RandomSeed))
	


