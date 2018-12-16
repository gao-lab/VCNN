import numpy as np
import pandas as pd
import h5py
import random
import glob
from sklearn.model_selection import StratifiedKFold
import os
# from ushuffle import Shuffler, shuffle
import pdb

"""
这个脚本是把序列转化为可训练的matrix的
"""


"""
sequence to matrix
"""
def seq_to_matrix(seq,seq_matrix,seq_order):
    '''
    change target 3D tensor according to sequence and order
    :param seq: 输入的单根序列
    :param seq_matrix: 输入的初始化的矩阵
    :param seq_order:这是第几个序列
    :return:
    '''
    for i in range(len(seq)):
        if((seq[i]=='A')|(seq[i]=='a')):
            seq_matrix[seq_order,i,0]=1
        if((seq[i]=='C')|(seq[i]=='c')):
            seq_matrix[seq_order,i,1]=1
        if((seq[i]=='G')|(seq[i]=='g')):
            seq_matrix[seq_order,i,2]=1
        if((seq[i]=='T')|(seq[i]=='t')):
            seq_matrix[seq_order,i,3]=1
    return seq_matrix

def genarate_matrix_for_train(seq_shape,seq_series):
    """
    这个函数是用于生成一个有sequence组成的大矩阵。
    :param shape: (seq number, sequence_length, 4)
    :param seq_series: 由seq组成的一个dataframe格式的文件
    :return:seq
    """
    seq_matrix = np.zeros(seq_shape)
    for i in range(seq_series.shape[0]):
        seq_tem = seq_series[i]
        seq_matrix = seq_to_matrix(seq_tem, seq_matrix, i)
    return seq_matrix


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


def generate_dataset_matrix(file_path):
    """
    由filepath得到数据的矩阵
    :param file_path:
    :return:
    """
    filenames = glob.glob(file_path+"/*.data")
    for allFileFa in filenames:
        AllTem = allFileFa.split("/")[-1].split(".")[0]

        output_dir = allFileFa.split(AllTem)[0].replace("motif_discovery", "HDF5")
        # 阳性集合
        SeqLen = 101
        ChipSeqlFileFa = pd.read_csv(allFileFa, sep=' ', header=None, index_col=None)
        seq_series = np.asarray(ChipSeqlFileFa.ix[:, 1])
        # for i in range(seq_series.shape[0]):
        #     SeqLen = max(SeqLen, len(seq_series[i]))
        seq_name = np.asarray(ChipSeqlFileFa.ix[:, 0]).astype("string")
        seq_matrix_out = genarate_matrix_for_train((seq_series.shape[0], SeqLen, 4), seq_series)
        seq_label_out = np.asarray(ChipSeqlFileFa.ix[:, 2])
        mkdir(output_dir)
        f = h5py.File(output_dir + AllTem +".hdf5")
        f.create_dataset("sequences",data=seq_matrix_out)
        f.create_dataset("labs",data=seq_label_out)
        f.create_dataset("seq_idx",data=seq_name)
        f.close()
        print(output_dir)


if __name__ == '__main__':
    base = {0:"A",1:"C",2:"G",3:"T"}
    allFileFaList = glob.glob("cnn.csail.mit.edu/motif_discovery/wgEncodeAwg*")

    for FilePath in allFileFaList:
        generate_dataset_matrix(FilePath)
