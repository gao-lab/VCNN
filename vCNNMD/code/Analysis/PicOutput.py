# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import glob
import h5py
import time
import math
import pickle


import pandas as pd
import sys
import math
from datetime import datetime
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

plt.switch_backend('agg')


def tictoc():

    return datetime.now().minute + datetime.now().second + datetime.now().microsecond*(10**-6)

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False
    


def AnalysisAUCMedian(filename, percentile=0):
    """

    :param filename:
    :return:
    """
    outputPath = RootPath+"OutputAnalysis/picTure/Simple/" + filename +"/"
    narrowPeak = RootPath + "./ChipSeqPeak/" + filename + ".narrowPeak"
    Peakfile = pd.read_csv(narrowPeak, sep="\t", header=None)
    FastaShape = Peakfile.shape[0]
    f = h5py.File(RootPath+"OutputAnalysis/AUChdf5/"+filename+".hdf5","r")
    mkdir(outputPath)

    BestCisFinder = 1000
    BestCisFinderCluster = 1000
    BestVCNNMD = 1000
    BestDreme= 1000



    KeyCisFinder = []
    KeyCisFinderCluster = []
    KeyVCNN = []
    KeyDreme= []
    CisFinderShape = 0
    CisFinderClusterShape = 0
    VCNNMDShape = 0
    VCNNBShape = 0
    CNNMDShape = 0
    CNNBShape = 0
    DremeShape = 0

    BestCisFinder = 0
    BestCisFinderCluster = 0
    BestCNNMD = 0
    BestVCNNB = 0
    BestCNNB = 0
    BestDreme = 0
    percentile = percentile/100.0

    for key in f.keys():
        if key[:9] == "VCNNMDOut":
            MotifTem = np.where(f[key].value<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > percentile and MotifTem < BestVCNNMD:
                BestVCNNMD = MotifTem
                KeyVCNN = f[key].value
            if MotifTem == BestVCNNMD:
                VCNNMDShape = f[key].shape[0]*1.0/FastaShape
        elif key[:8] == "CNNMDOut":
            MotifTem = np.where(f[key].value<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestCNNMD and MotifTem > percentile:
                BestCNNMD = MotifTem
            if MotifTem == BestCNNMD:
                CNNMDShape = f[key].shape[0]*1.0/FastaShape

        elif key[:8] == "VCNNBOut":
            MotifTem = np.where(f[key].value<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestVCNNB and MotifTem > percentile:
                BestVCNNB = MotifTem
            if MotifTem == BestVCNNB :
                VCNNBShape = f[key].shape[0]*1.0/FastaShape

        elif key[:7] == "CNNBOut":
            MotifTem = np.where(f[key].value<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestCNNB and MotifTem > percentile:
                BestCNNB = MotifTem
            if MotifTem == BestCNNB:
                CNNBShape = f[key].shape[0] * 1.0 / FastaShape

        elif key[:8] == "DremeOut":
            MotifTem = np.where(f[key].value<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > percentile and MotifTem < BestDreme:
                BestDreme = MotifTem
                KeyDreme = f[key].value
            if MotifTem == BestDreme:
                DremeShape = f[key].shape[0]*1.0/FastaShape

        elif key[:19] == "CisFinderClusterOut":
            MotifTem = np.where(f[key].value<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > percentile and MotifTem < BestCisFinderCluster:
                BestCisFinderCluster = MotifTem
                KeyCisFinderCluster = f[key].value
            if MotifTem == BestCisFinderCluster:
                CisFinderClusterShape = f[key].shape[0]*1.0/FastaShape

        else:
            MotifTem = np.where(f[key].value<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > percentile and MotifTem < BestCisFinder:
                BestCisFinder = MotifTem
                KeyCisFinder = f[key].value
            if MotifTem == BestCisFinder:
                CisFinderShape = f[key].shape[0]*1.0/FastaShape

    ax = plt.subplot()  # 创建作图区域
    # 蓝色矩形的红线：50%分位点是4.5,上边沿：25%分位点是2.25,下边沿：75%分位点是6.75
    ax.boxplot([KeyVCNN, KeyDreme, KeyCisFinder, KeyCisFinderCluster])
    ax.set_xticklabels(['VCNNMD', 'Dreme','CisFinder','CisFinderCluster'])
    plt.xlabel(filename)  # X轴标签
    plt.savefig(outputPath+"./"+ str(percentile)+"boxres.png")
    plt.close()


    print(CisFinderShape)
    print(VCNNMDShape)
    print(DremeShape)
    print(CisFinderClusterShape)

    return VCNNMDShape,VCNNBShape,CNNMDShape,CNNBShape,DremeShape,CisFinderShape,CisFinderClusterShape

def AnalysisAccracy(filename, percentile=0):
    """

    :param filename:
    :return:
    """
    outputPath = RootPath+"OutputAnalysis/picTure/Simple/" + filename +"/"
    narrowPeak = RootPath + "./ChipSeqPeak/" + filename + ".narrowPeak"
    Peakfile = pd.read_csv(narrowPeak, sep="\t", header=None)
    FastaShape = Peakfile.shape[0]
    f = h5py.File(RootPath+"OutputAnalysis/AUChdf5/"+filename+".hdf5","r")
    mkdir(outputPath)

    # 回帖最靠中心的motif的均值
    BestCisFinder = 0
    BestCisFinderCluster = 0
    BestVCNNMD = 0
    BestCNNMD = 0
    BestVCNNB = 0
    BestCNNB = 0
    BestDreme= 0
    BestMemeChip = 0


    for key in f.keys():

        if key[:9] == "VCNNMDOut":
            MotifTem = np.where(f[key].value<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestVCNNMD:
                BestVCNNMD = MotifTem

        elif key[:8] == "CNNMDOut":
            MotifTem = np.where(f[key].value<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestCNNMD:
                BestCNNMD = MotifTem

        elif key[:8] == "VCNNBOut":
            MotifTem = np.where(f[key].value<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestVCNNB:
                BestVCNNB = MotifTem

        elif key[:7] == "CNNBOut":
            MotifTem = np.where(f[key].value<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestCNNB:
                BestCNNB = MotifTem

        elif key[:8] == "DremeOut":
            MotifTem = np.where(f[key].value<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestDreme:
                BestDreme = MotifTem

        elif key[:19] == "CisFinderClusterOut" and int(key[-3:])<3:
            MotifTem = np.where(f[key].value<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestCisFinderCluster:
                BestCisFinderCluster = MotifTem
        elif key[:12] == "CisFinderOut":
            MotifTem = np.where(f[key].value<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestCisFinder:
                BestCisFinder = MotifTem
        elif key[:11] == "MemeChipOut":
            MotifTem = np.where(f[key].value<100)[0].shape[0]/(FastaShape*1.0)
            if MotifTem > BestMemeChip:
                BestMemeChip = MotifTem
    BestCisFinderCluster = BestCisFinder
    return BestVCNNMD, BestCNNMD, BestVCNNB, BestCNNB, BestDreme, BestCisFinder, BestCisFinderCluster,BestMemeChip




def AccuracyRatio(RootPath, AlldataPath,percentile=1):

    BestVCNNBlist = []
    BestCNNBlist = []
    BestDremelist = []
    BestCisFinderlist = []
    BestCisFinderClusterlist = []
    BestMemeChiplist = []


    for file in AlldataPath:
        filename = file.split("/")[-1].replace(".narrowPeak","")
        print(file)

        (BestVCNNMD, BestCNNMD, BestVCNNB, BestCNNB, BestDreme,
         BestCisFinder, BestCisFinderCluster,BestMemeChip)= AnalysisAccracy(filename,percentile)

        BestVCNNBlist.append(BestVCNNB)
        BestCNNBlist.append(BestCNNB)
        BestDremelist.append(BestDreme)
        BestCisFinderlist.append(BestCisFinder)
        BestCisFinderClusterlist.append(BestCisFinderCluster)
        BestMemeChiplist.append(BestMemeChip)
    # 蓝色矩形的红线：50%分位点是4.5,上边沿：25%分位点是2.25,下边沿：75%分位点是6.75
    dictlist = {}
    namelist = ['vCNN-based model', 'CNN-based model', 'DREME','CisFinder','MEME-ChIP']

    resultlist = [BestVCNNBlist, BestCNNBlist,BestDremelist,
                  BestCisFinderClusterlist,BestMemeChiplist]
    for i in range(len(namelist)):
        dictlist[namelist[i]] = resultlist[i]
    Pddict = pd.DataFrame(dictlist)
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=Pddict)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(RootPath+"/OutputAnalysis/picTure/boxres.png")
    plt.close()


    BestVCNNBArray = np.asarray(BestVCNNBlist)
    BestCNNBArray = BestVCNNBArray - np.asarray(BestCNNBlist)
    BestDremeArray = BestVCNNBArray - np.asarray(BestDremelist)
    BestCisFinderClusterArray = BestVCNNBArray - np.asarray(BestCisFinderClusterlist)
    BestMemeChipArray = BestVCNNBArray - np.asarray(BestMemeChiplist)
    dictlist = {}
    namelist = ['vCNN-based model', 'CNN-based model', 'DREME','CisFinder','MEME-ChIP']

    resultlist = [BestVCNNBArray -BestVCNNBArray, BestCNNBArray,
                BestDremeArray, BestCisFinderClusterArray,BestMemeChipArray]
    for i in range(2, len(namelist)):
        dictlist[namelist[i]] = resultlist[i]
        print(namelist[i]," ",resultlist[i][resultlist[i] >= 0].shape[0])
    Pddict = pd.DataFrame(dictlist)

    # plt.figure(figsize=(6, 4))
    Barplot = sns.barplot(data=Pddict, capsize=.2)
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    Barplot.set_xticklabels(
        Barplot.get_xticklabels(),
        # rotation=45,
        # horizontalalignment='right',
        # fontweight='light',
        fontsize=15
    )
    # plt.ylabel("Improvement of AUC by \n vCNN-based motif discovery", fontsize=15)  # X轴标签
    plt.title("Improvement of AUC by \n vCNN-based motif discovery", fontsize=15)  # X轴标签
    plt.tight_layout()
    plt.savefig(RootPath+"/OutputAnalysis/picTure/Resboxres.png")
    plt.close()
    
    f =open("./sta.txt")
    pvalue = stats.mannwhitneyu(BestCNNBlist, BestVCNNBlist, alternative="less")[1]
    f.writelines(str(pvalue))

if __name__ == '__main__':
    
    RootPath = "../../"
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    CTCFfiles = glob.glob(RootPath+"/Peak/"+"*Ctcf*")
    AccuracyRatio(RootPath, CTCFfiles)