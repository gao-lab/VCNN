# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import glob
import h5py
import time
import math
import pickle
import pdb
import pandas as pd
import sys
import math
from datetime import datetime

def tictoc():

    return datetime.now().minute + datetime.now().second + datetime.now().microsecond*(10**-6)

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

    
def cisFinder(InputFile, fileName):
    """
    调用cisFinder,
    :param InputFile:  fasta格式的文件
    :return:
    """
    softwarePath = "/home/lijy/CisFinder/patternFind"
    outputDir = "/home/lijy/CisFinder/Output/" + fileName + "/Result.txt"
    mkdir("/home/lijy/CisFinder/Output/" + fileName)
    tmp_cmd = softwarePath + " -i "+ InputFile + " " + "-o "+ outputDir
    os.system(tmp_cmd)



def cisFinderCluster():
    """
    把每个CisFinder的数据做聚类
    :param InputFile:
    :param motifFile:
    :param fileName:
    :return:
    """

    # motif路径
    CisFinderMotifPath = "/home/lijy/VCNN_VS_Classical/Cisfinder/result/" + filename + "/Result.txt"

    # 输出路径
    CisFinderOutputPath = "/home/lijy/VCNN_VS_Classical/Cisfinder/result/" + filename + "/Cluster/"
    mkdir(CisFinderOutputPath)

    # 调用cisfinder找peak点

    softwarePath = "/home/lijy/CisFinder/patternCluster"
    tmp_cmd = softwarePath + " -i "+ CisFinderMotifPath + " -o "+ CisFinderOutputPath + "CisfinerMotif.txt"
    os.system(tmp_cmd)




if __name__ == '__main__':
    
    CTCFfiles = glob.glob("/home/lijy/chip-seqFa/"+"*Ctcf*")
    # TOObigForCisfinder = open("/home/lijy/CisFinder/"+"/bigData.txt", "w")

    for file in CTCFfiles:
        filePath = file
        filename = file.split("/")[-1].split(".")[0]
        print(file)
        cisFinder(filePath, filename)
        cisFinderCluster()
