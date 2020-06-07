import pandas as pd
import numpy as np
import glob
import os
import pdb


fileNameList = glob.glob("/home/lijy/chip-seq/wgEncodeAwgTfbs*")

for fileName in fileNameList:
    outFileName = fileName.split("/")[-1].split(".")[0] + ".fa"
    tmp_cmd = str("bedtools getfasta -fi /home/jiangs/jiangs/postgraduate/gene_model/stage_5/04_gencode_refseq_lncRNAdb_v5/25_other_work_CDS/hg19.fa -bed "
                  + fileName + " -fo " + " /home/lijy/chip-seqFa/" + outFileName)
    os.system(tmp_cmd)

def mkdir(path):
    """
    创建目录
    :param path:目录的路径
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

def extractPeak(fileNameList):

    for filePath in fileNameList:
        fileTem = np.asarray(pd.read_csv(filePath, sep="\t", header=None)[9])
        fileOutpath= filePath.replace("chip-seq","chipSeqPeak")
        np.savetxt(fileOutpath, fileTem)


def CountSeqDataSetLen(fileNameList):
    DataLenlist = []
    NameList = []
    for filePath in fileNameList:
        fileTem1 = np.asarray(pd.read_csv(filePath, sep="\t", header=None)[1])
        fileTem2 = np.asarray(pd.read_csv(filePath, sep="\t", header=None)[2])
        DataLenlist.append(np.sum(fileTem2-fileTem1))
        NameList.append(filePath.split("/")[-1].split(".")[0])
    Outdict = {"Name":NameList, "DataLen":DataLenlist}
    OutDataFrame = pd.DataFrame(Outdict)
    OutDataFrame.to_csv("/home/lijy/ChipSeqDataLen.csv")

def CountSeqNum(filelist, OutputPath):
    """
    
    :param filelist:
    :param OutputPath:
    :return:
    """
    DataLenlist = []
    NameList = []
    for file in filelist:
        f = open(file,"r")
        length = len(f.readlines())
        Name = file.split("/")[-1].replace(".narrowPeak","")
        NameList.append(Name)
        DataLenlist.append(length)
    Outdict = {"Name":NameList, "DataLen":DataLenlist}
    OutDataFrame = pd.DataFrame(Outdict)
    OutDataFrame.to_csv(OutputPath+"/ChipSeqDataNum.csv")
    
    
if __name__ == '__main__':
    PeakPath = "../../TFBSEnconde/chipSeq/"
    filelist = glob.glob(PeakPath+"/*.narrowPeak")
    CountSeqNum(filelist, "./")
    
    TestLenlist=['wgEncodeAwgTfbsSydhHelas3Brf1UniPk',
       'wgEncodeAwgTfbsHaibA549GrPcr1xDex500pmUniPk',
       'wgEncodeAwgTfbsSydhHelas3Prdm19115IggrabUniPk',
       'wgEncodeAwgTfbsHaibHepg2Cebpdsc636V0416101UniPk',
       'wgEncodeAwgTfbsSydhK562Mafkab50322IggrabUniPk',
       'wgEncodeAwgTfbsBroadH1hescRbbp5a300109aUniPk',
       'wgEncodeAwgTfbsSydhHuvecCfosUcdUniPk']
    
    Testfilelist = [PeakPath+Name+".narrowPeak" for Name in TestLenlist]
    CountSeqNum(Testfilelist, "../")

    
    """
    length = pd.read_csv("./ChipSeqDataLen.csv", index_col=0)
    DataLenArray = np.asarray(length['DataLen'])
    DataNameArray = np.asarray(length['Name'])
    np.percentile(DataLenArray, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    LenSort = np.argsort(DataLenArray)
    indextem =[0,20,120,240,360,600,680]
    TestLenlist=['wgEncodeAwgTfbsSydhHelas3Brf1UniPk',
       'wgEncodeAwgTfbsHaibA549GrPcr1xDex500pmUniPk',
       'wgEncodeAwgTfbsSydhHelas3Prdm19115IggrabUniPk',
       'wgEncodeAwgTfbsHaibHepg2Cebpdsc636V0416101UniPk',
       'wgEncodeAwgTfbsSydhK562Mafkab50322IggrabUniPk',
       'wgEncodeAwgTfbsBroadH1hescRbbp5a300109aUniPk',
       'wgEncodeAwgTfbsSydhHuvecCfosUcdUniPk']
    """