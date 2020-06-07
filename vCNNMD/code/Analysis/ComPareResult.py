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
    
    
def VCNNFileIntoCisfinDer():
    """
    
    :return:
    """
    AlldataPath = glob.glob(RootPath+"VCNNMD/result/wgEncodeAwgTfbs*")
    

    for files in AlldataPath:
    
    
        filename = files.split("/")[-1]
    
    
        motif_path = RootPath+"/VCNNMD/result/" + filename + "/recover_PWM/*.txt"
        Motifs = glob.glob(motif_path)

        title = open(RootPath+"/title/title.txt", 'rU')

        f = open(files + "/" + "VCNNMotif.txt", "w")

        for count, line in enumerate(title):
            if count <2:
                # f.write(line)
                pass
            else:
                MotifTitle = line

        motifSeqNumlist = []
        for motif in Motifs:
            motifSeqNum = int(motif.split("/")[-1].replace(".txt","").split("_")[-1])
            motifSeqNumlist.append(motifSeqNum)
        motifSeqNumlist.sort()

        try:
            NumThreshold = motifSeqNumlist[-min(3, len(motifSeqNumlist))]
        except:
            import pdb
            pdb.set_trace()
        
        
        for i, motif in enumerate(Motifs):
            motifSeqNum = int(motif.split("/")[-1].replace(".txt","").split("_")[-1])
            # MotifTitleTem = MotifTitle[:4] + str(i) + MotifTitle[5:8] + filename[15:]
            MotifTitleTem = MotifTitle[:4] + str(i)

            if motifSeqNum >= NumThreshold:
    
                kernel = np.loadtxt(motif)
                
                f.write(MotifTitleTem+"\n")
            
                for i in range(kernel.shape[0]):
                    Column = 0
                    f.write(str(i) + "\t")

                    for j in range(3):
                        f.write(str(int(kernel[i, j]*100)) + "\t")
                        Column = Column + int(kernel[i, j]*100)
                    f.write(str(100 - Column) + "\t")
                    f.write("\n")
                f.write("\n")

        f.close()
                
def dremeFileIntoCisFinder():
    """
    调用dreme,
    :param InputFile: fasta格式的文件
    :return:
    """
    def fileProcess(line, Num):
        A = int(np.float64(line[:8])*100)
        C = int(np.float64(line[9:17])*100)
        G = int(np.float64(line[18:26])*100)
        T = 100 - A - C - G
        liubai = str(Num) + "\t"
        lineOut = liubai +  str(A) + "\t" + str(C) + "\t" + str(G) + "\t" + str(T) + "\n"
        return lineOut

    AlldataPath = glob.glob(RootPath+"Dreme/result/wgEncodeAwgTfbs*")

    for files in AlldataPath:
    
        filename = files.split("/")[-1]
    
        motif_path = RootPath+"/Dreme/result/" + filename + "/dreme.txt"
        Num = 0
        if os.path.isfile(motif_path):
            f = open(files + "/" + "DremeMotif.txt", "w")
            LineIsMotif = False
            MotifNum = 0
            for count, line in enumerate(open(motif_path, 'rU')):
                if LineIsMotif:
                    if line == "\n":
                        f.write(line)
                    else:
                        f.write(fileProcess(line, Num))
                        Num = Num + 1
                if line[:25]=="letter-probability matrix":
                    Num = 0
                    LineIsMotif = True
                    MotifNum = MotifNum + 1
                    if MotifNum > 3:
                        break
                    f.write(">dreme"+ str(MotifNum) + "\n")
                elif line=="\n":
                    LineIsMotif = False
                if MotifNum>3:
                    break

        else:
            print("wrong:"+motif_path)
    






def Accuracy(filename, CisFinderU=False, CisFinderClusterU=False, DremeU=False,VCNNMDU=False):
    """
    
    :param InputFile:
    :param fileName:
    :return:
    """
    def PeakHandle(narrowPeak):
        """
        提取每条序列的位置
        :param narrowPeak:
        :return:
        """
        peakDict = {}
        Peakfile = pd.read_csv(narrowPeak, sep="\t", header=None)
        for i in range(Peakfile.shape[0]):
            Name = Peakfile[0][i]+":" + str(Peakfile[1][i])+ "-" + str(Peakfile[2][i])
            Peak = Peakfile[9][i]
            peakDict[Name] = Peak
        return peakDict
        
    def CisFinder(CisFinderPath,peakDict):
        """
        find the peack point on sequences compared to the motif
        :param narrowPeak:
        :return:
        """
        if os.path.isfile(CisFinderPath + "./Peak.txt"):
            MotifPeakOut = {"M001":[],"M002":[],"M003":[]}
            MotifPeakOutDict = {"M001":{},"M002":{},"M003":{}}

            Peak = pd.read_csv(CisFinderPath + "/Peak.txt", sep="\t", skiprows=1)
            SeqName = Peak["MotifName"]
            PeakMotif = Peak["Len"] + Peak["Strand"]/2
            MotifName = np.asarray(Peak["Headers:"])
            Score = np.asarray(Peak["Start"])
            ######三个motif##########
            M001 = np.where(MotifName == "M001")[0]


            for i in range(M001.shape[0]):
                name = SeqName[M001[i]]
                Tem = abs(PeakMotif[M001[i]] - peakDict[name])
                scoretem = Score[i]
                if name not in MotifPeakOutDict["M001"].keys():
                    MotifPeakOutDict["M001"][name] = scoretem

                    MotifPeakOut["M001"].append(Tem)
                elif scoretem > MotifPeakOutDict["M001"][name]:
                    MotifPeakOut["M001"][-1] =Tem

            M002 = np.where(MotifName == "M002")[0]

            for i in range(M002.shape[0]):
                name = SeqName[M002[i]]
                Tem = abs(PeakMotif[M002[i]] - peakDict[name])
                scoretem = Score[i]
                if name not in MotifPeakOutDict["M002"].keys():
                    MotifPeakOutDict["M002"][name] = scoretem

                    MotifPeakOut["M002"].append(Tem)
                elif scoretem > MotifPeakOutDict["M002"][name]:
                    MotifPeakOut["M002"][-1] =Tem
    
            M003 = np.where(MotifName == "M003")[0]
            for i in range(M003.shape[0]):
                name = SeqName[M003[i]]
                Tem = abs(PeakMotif[M003[i]] - peakDict[name])
                scoretem = Score[i]
                if name not in MotifPeakOutDict["M003"].keys():
                    MotifPeakOutDict["M003"][name] = scoretem

                    MotifPeakOut["M003"].append(Tem)
                elif scoretem > MotifPeakOutDict["M003"][name]:
                    MotifPeakOut["M003"][-1] = Tem

            return MotifPeakOut
        else:
            return {}

    def CisFinderCluster(CisFinderClusterPath,peakDict):
        """
        find the peack point on sequences compared to the motif
        :param narrowPeak:
        :return:
        """
        if os.path.isfile(CisFinderClusterPath + "/Peak.txt"):

            Peak = pd.read_csv(CisFinderClusterPath+"./Peak.txt", sep="\t", skiprows=1)
            SeqName = np.asarray(Peak["MotifName"])
            PeakMotif = np.asarray(Peak["Len"]) + np.asarray(Peak["Strand"]) / 2
            MotifName = Peak["Headers:"]
            Score = np.asarray(Peak["Start"])

            MotifNameSet = list(set(MotifName))
            MotifNameSet.remove("NONE")
            MotifPeakOut = {}
            MotifPeakOutDict = {}
            for name in MotifNameSet:
                MotifPeakOut[name] = []
                MotifPeakOutDict[name] = {}

            for mname in MotifNameSet:
                MTem = np.where(MotifName == mname)[0]
                for i in range(MTem.shape[0]):
                    name = SeqName[MTem[i]]
                    Tem = abs(PeakMotif[MTem[i]] - peakDict[name])
                    scoretem = Score[i]
                    if name not in MotifPeakOutDict[mname].keys():
                        MotifPeakOutDict[mname][name] = scoretem

                        MotifPeakOut[mname].append(Tem)
                    elif scoretem > MotifPeakOutDict[mname][name]:
                        MotifPeakOut[mname][-1]= Tem


            return MotifPeakOut
        else:
            return {}

    def Dreme(DremePath,peakDict):
        """
        find the peack point on sequences compared to the motif
        :param narrowPeak:
        :return:
        """
        if os.path.isfile(DremePath + "/Peak.txt"):

            Peak = pd.read_csv(DremePath+"./Peak.txt", sep="\t", skiprows=1)
            SeqName = np.asarray(Peak["MotifName"])
            PeakMotif = np.asarray(Peak["Len"]) + np.asarray(Peak["Strand"]) / 2
            MotifName = Peak["Headers:"]
            Score = np.asarray(Peak["Start"])

            MotifNameSet = list(set(MotifName))
            MotifNameSet.remove("NONE")
            MotifPeakOut = {}
            MotifPeakOutDict = {}
            for name in MotifNameSet:
                MotifPeakOut[name] = []
                MotifPeakOutDict[name] = {}

            for mname in MotifNameSet:
                MTem = np.where(MotifName == mname)[0]
                for i in range(MTem.shape[0]):
                    name = SeqName[MTem[i]]
                    Tem = abs(PeakMotif[MTem[i]] - peakDict[name])
                    scoretem = Score[i]
                    if name not in MotifPeakOutDict[mname].keys():
                        MotifPeakOutDict[mname][name] = scoretem

                        MotifPeakOut[mname].append(Tem)
                    elif scoretem > MotifPeakOutDict[mname][name]:
                        MotifPeakOut[mname][-1]= Tem


            return MotifPeakOut
        else:
            return {}
        
    def VCNNMD(VCNNMDPath,peakDict):
        """
        find the peack point on sequences compared to the motif
		:param narrowPeak:
		:return:
		"""
        if os.path.isfile(VCNNMDPath + "/Peak.txt"):

            Peak = pd.read_csv(VCNNMDPath+"./Peak.txt", sep="\t", skiprows=1)
            SeqName = np.asarray(Peak["MotifName"])
            PeakMotif = np.asarray(Peak["Len"]) + np.asarray(Peak["Strand"]) / 2
            MotifName = Peak["Headers:"]
            Score = np.asarray(Peak["Start"])

            MotifNameSet = list(set(MotifName))
            MotifNameSet.remove("NONE")
            MotifPeakOut = {}
            MotifPeakOutDict = {}
            for name in MotifNameSet:
                MotifPeakOut[name] = []
                MotifPeakOutDict[name] = {}

            for mname in MotifNameSet:
                MTem = np.where(MotifName == mname)[0]
                for i in range(MTem.shape[0]):
                    name = SeqName[MTem[i]]
                    Tem = abs(PeakMotif[MTem[i]] - peakDict[name])
                    scoretem = Score[i]
                    if name not in MotifPeakOutDict[mname].keys():
                        MotifPeakOutDict[mname][name] = scoretem

                        MotifPeakOut[mname].append(Tem)
                    elif scoretem > MotifPeakOutDict[mname][name]:
                        MotifPeakOut[mname][-1] = Tem

            return MotifPeakOut

        else:
            return {}


    narrowPeak = RootPath + "./ChipSeqPeak/" + filename + ".narrowPeak"
    
    CisFinderPath = RootPath+"/Peak/" + filename + "/CisFinder/"
    CisFinderClusterPath = RootPath+"/Peak/" + filename + "/CisFinderCluster/"
    DremePath = RootPath+"/Peak/" + filename + "/Dreme/"
    VCNNMDPath = RootPath+"/Peak/" + filename + "/VCNNMD/"
    CNNMDPath = RootPath+"/Peak/" + filename + "/CNNMD/"
    VCNNBPath = RootPath+"/Peak/" + filename + "/VCNNB/"
    CNNBPath = RootPath+"/Peak/" + filename + "/CNNB/"


    peakDict = PeakHandle(narrowPeak)

    if CisFinderU:
        CisFinderOut = CisFinder(CisFinderPath, peakDict)
    if CisFinderClusterU:
        CisFinderClusterOut = CisFinderCluster(CisFinderClusterPath, peakDict)
    if DremeU:
        DremeOut = Dreme(DremePath, peakDict)
    if VCNNMDU:
        # VCNNMDOut = VCNNMD(VCNNMDPath, peakDict)
        CNNMDOut = VCNNMD(CNNMDPath, peakDict)
        # VCNNBOut = VCNNMD(VCNNBPath, peakDict)
        CNNBOut = VCNNMD(CNNBPath, peakDict)


    if CisFinderU and CisFinderClusterU and DremeU and VCNNMDU:

        f = h5py.File(RootPath+"/OutputAnalysis/AUChdf5/"+filename+".hdf5","w")
        for i in CisFinderOut.keys():
            f.create_dataset("CisFinderOut"+i, data=np.asarray(CisFinderOut[i]))
        for i in DremeOut.keys():
            f.create_dataset("DremeOut"+i, data=np.asarray(DremeOut[i]))
        for i in VCNNMDOut.keys():
            f.create_dataset("VCNNMDOut"+i, data=np.asarray(VCNNMDOut[i]))
        for i in CNNMDOut.keys():
            f.create_dataset("CNNMDOut" + i, data=np.asarray(CNNMDOut[i]))
        for i in VCNNBOut.keys():
            f.create_dataset("VCNNBOut" + i, data=np.asarray(VCNNBOut[i]))
        for i in CNNBOut.keys():
            f.create_dataset("CNNBOut" + i, data=np.asarray(CNNBOut[i]))
        for i in CisFinderClusterOut.keys():
            f.create_dataset("CisFinderClusterOut"+i, data=np.asarray(CisFinderClusterOut[i]))
        f.close()
        gc.collect()
    else:

        f = h5py.File(RootPath+"/OutputAnalysis/AUChdf5/"+filename+".hdf5","a")

        if CisFinderU:
            for i in CisFinderOut.keys():
                f.create_dataset("CisFinderOut" + i, data=np.asarray(CisFinderOut[i]))
        if CisFinderClusterU:
            for i in CisFinderClusterOut.keys():
                f.create_dataset("CisFinderClusterOut" + i, data=np.asarray(CisFinderClusterOut[i]))
        if DremeU:
            for i in DremeOut.keys():
                f.create_dataset("DremeOut" + i, data=np.asarray(DremeOut[i]))
        if VCNNMDU:
            for i in CNNMDOut.keys():
                try:
                    del f["CNNMDOut" + i]
                except:
                    pass
                try:
                    f.create_dataset("CNNMDOut" + i, data=np.asarray(CNNMDOut[i]))
                except:
                    pass

            for i in CNNBOut.keys():
                try:
                    del f["CNNBOut" + i]
                except:
                    pass
                try:
                    f.create_dataset("CNNBOut" + i, data=np.asarray(CNNBOut[i]))
                except:
                    pass
        f.close()
        gc.collect()



def AnalysisAUC(filename, percentile=0):
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

    # The average of the most central motif of the reply
    BestCisFinder = 1000
    BestCisFinderCluster = 1000
    BestVCNN = 1000
    BestDreme= 1000



    KeyCisFinder = []
    KeyCisFinderCluster = []
    KeyVCNN = []
    KeyDreme= []
    CisFinderShape = 0
    CisFinderClusterShape = 0
    VCNNShape = 0
    DremeShape = 0
    percentile = percentile/100.0

    for key in f.keys():

        if key[:7] == "VCNNMDOut":
            MotifTem = np.mean(f[key].value)
            if f[key].shape[0] > FastaShape * percentile and MotifTem < BestVCNN:
                BestVCNN = MotifTem
                KeyVCNN = f[key].value
            if MotifTem == BestVCNN:
                VCNNShape = f[key].shape[0]*1.0/FastaShape

        elif key[:8] == "DremeOut":
            MotifTem = np.mean(f[key].value)
            if f[key].shape[0] > FastaShape * percentile and MotifTem < BestDreme:
                BestDreme = MotifTem
                KeyDreme = f[key].value
            if MotifTem == BestDreme:
                DremeShape = f[key].shape[0]*1.0/FastaShape

        elif key[:19] == "CisFinderClusterOut":
            MotifTem = np.mean(f[key].value)
            if f[key].shape[0] > FastaShape * percentile and MotifTem < BestCisFinderCluster:
                BestCisFinderCluster = MotifTem
                KeyCisFinderCluster = f[key].value
            if MotifTem == BestCisFinderCluster:
                CisFinderClusterShape = f[key].shape[0]*1.0/FastaShape

        else:
            MotifTem = np.mean(f[key].value)
            if f[key].shape[0] > FastaShape * percentile and MotifTem < BestCisFinder:
                BestCisFinder = MotifTem
                KeyCisFinder = f[key].value
            if MotifTem == BestCisFinder:
                CisFinderShape = f[key].shape[0]*1.0/FastaShape

    ax = plt.subplot()  # 创建作图区域
    # 蓝色矩形的红线：50%分位点是4.5,上边沿：25%分位点是2.25,下边沿：75%分位点是6.75
    ax.boxplot([KeyVCNN, KeyDreme, KeyCisFinder, KeyCisFinderCluster])
    ax.set_xticklabels(['VCNNMD', 'Dreme','CisFinder','CisFinderCluster'])
    plt.xlabel(filename)  # X轴标签
    plt.savefig(outputPath+"./boxres.png")
    plt.close()
    if BestVCNN == 1000:
        BestVCNN = 100
    if BestDreme == 1000:
        BestDreme = 100
    if BestCisFinder == 1000:
        BestCisFinder = 100
    if BestCisFinderCluster == 1000:
        BestCisFinderCluster = 100

    print(CisFinderShape)
    print(VCNNShape)
    print(DremeShape)
    print(CisFinderClusterShape)

    return BestVCNN, BestDreme, BestCisFinder, BestCisFinderCluster, VCNNShape,DremeShape,CisFinderShape,CisFinderClusterShape

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

    #The average of the most central motif of the reply
    BestCisFinder = 1000
    BestCisFinderCluster = 1000
    BestVCNNMD = 1000
    BestDreme= 1000

    # The average of the most central motif of the reply


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

    return BestVCNNMD, BestCNNMD, BestVCNNB, BestCNNB, BestDreme, BestCisFinder, BestCisFinderCluster




def AccuracyRatio(RootPath, AlldataPath,percentile=1):

    BestVCNNBlist = []
    BestCNNBlist = []
    BestDremelist = []
    BestCisFinderlist = []
    BestCisFinderClusterlist = []


    for file in AlldataPath:
        filename = file.split("/")[-1].replace(".narrowPeak","")
        print(file)

        (BestVCNNMD, BestCNNMD, BestVCNNB, BestCNNB, BestDreme,
         BestCisFinder, BestCisFinderCluster)= AnalysisAccracy(filename,percentile)

        BestVCNNBlist.append(BestVCNNB)
        BestCNNBlist.append(BestCNNB)
        BestDremelist.append(BestDreme)
        BestCisFinderlist.append(BestCisFinder)
        BestCisFinderClusterlist.append(BestCisFinderCluster)
    ax = plt.subplot()
    dictlist = {}
    namelist = ['vCNN-based model', 'CNN-based model', 'DREME','CisFinder','MEME-ChIP']

    resultlist = [BestVCNNBlist, BestCNNBlist,BestDremelist,
                  BestCisFinderClusterlist]
    for i in range(len(namelist)):
        dictlist[namelist[i]] = resultlist[i]
    Pddict = pd.DataFrame(dictlist)
    sns.boxplot(data=Pddict)

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xlabel("ctcf")  # X轴标签
    plt.ylabel("Accuracy")  # X轴标签
    plt.savefig(RootPath+"/OutputAnalysis/picTure/boxres.png")
    plt.close()


    BestVCNNBArray = np.asarray(BestVCNNBlist)
    BestCNNBArray = BestVCNNBArray - np.asarray(BestCNNBlist)
    BestDremeArray = BestVCNNBArray - np.asarray(BestDremelist)
    BestCisFinderArray = BestVCNNBArray - np.asarray(BestCisFinderlist)
    BestCisFinderClusterArray = BestVCNNBArray - np.asarray(BestCisFinderClusterlist)
    dictlist = {}
    namelist = ['vCNN-based model', 'CNN-based model', 'DREME','CisFinder','MEME-ChIP']

    resultlist = [BestVCNNBArray -BestVCNNBArray, BestCNNBArray,
                BestDremeArray, BestCisFinderClusterArray]
    for i in range(len(namelist)):
        dictlist[namelist[i]] = resultlist[i]
        print(namelist[i]," ",resultlist[i][resultlist[i] >= 0].shape[0])
    Pddict = pd.DataFrame(dictlist)

    ax = plt.subplot()
    sns.boxplot(data=Pddict)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xlabel("ctcf")  # X轴标签
    plt.ylabel("Accuracy res with vCNN-based model")  # X轴标签
    plt.savefig(RootPath+"/OutputAnalysis/picTure/Resboxres.png")
    plt.close()



if __name__ == '__main__':
    RootPath = "../../"

    CTCFfiles = glob.glob(RootPath+"/Peak/"+"*Ctcf*")
    for file in CTCFfiles:
        filePath = file
        filename = file.split("/")[-1].replace("narrowPeak","")
        print(file)
        Accuracy(filename, VCNNMDU=True)
    AccuracyRatio(RootPath, CTCFfiles)
