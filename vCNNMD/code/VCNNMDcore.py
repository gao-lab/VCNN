# -*- coding: utf-8 -*-
import time
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.warn("deprecated", DeprecationWarning)
    print("stop warning")
import os
import numpy as np
from build_models import *
from  seq_to_matrix import *
import glob
import pdb
from keras import backend as K
from datetime import datetime
def tictoc():
    return datetime.now().minute * 60 + datetime.now().second + datetime.now().microsecond*(10**-6)
import gc
from sklearn import mixture
import scipy.stats


###############vCNN based model######################################

def recover_ker(model, modeltype, KernelIndex=0):
    """
    vCNN_lg还原mask
    :param resultPath:
    :param modeltype:
    :param input_shape:
    :return:
    """
    try:
        KernelIndex.shape
    except:
        KernelIndex = range(K.get_value(model.layers[0].kernel).shape[2])



    def CutKerWithMask(MaskArray, KernelArray):

        CutKernel = []
        for Kid in range(KernelArray.shape[-1]):
            MaskTem = MaskArray[:, :, Kid].reshape(2, )
            leftInit = int(round(max(MaskTem[0] - 3, 0), 0))
            rightInit = int(round(min(MaskTem[1], KernelArray.shape[0] - 1), 0))
            if rightInit - leftInit >= 5:
                kerTem = KernelArray[leftInit:rightInit, :, Kid]
                CutKernel.append(kerTem)
        return CutKernel

    # 重新load模型
    if modeltype == "CNN":
        kernelTem = K.get_value(model.layers[0].kernel)[:,:,KernelIndex]
        kernel = []
        for i in range(kernelTem.shape[2]):
            kernel.append(kernelTem[:,:,i])

    elif modeltype == "vCNN":
        k_weights = K.get_value(model.layers[0].k_weights)[:,:,KernelIndex]
        kernelTem = K.get_value(model.layers[0].kernel)[:,:,KernelIndex]
        kernel = CutKerWithMask(k_weights, kernelTem)
    else:
        kernel = model.layers[0].get_kernel()[:,:,KernelIndex] * model.layers[0].get_mask()[:,:,KernelIndex]
    return kernel

def NormPwm(seqlist, Cut=False):
    """
    传入seqlist返回sequence构成的motif
    :param seqlist:
    :return:
    """
    SeqArray = np.asarray(seqlist)
    Pwm = np.sum(SeqArray,axis=0)
    Pwm = Pwm / Pwm.sum(axis=1, keepdims=1)
    
    if not Cut:
        return Pwm

    return Pwm

def KSselect(KernelSeqs):
    """
    针对选出来的kernel进行对应序列筛选
    :param KSconvValue:
    :param KernelSeqs:
    :return:
    """
    ########使用高斯混合模型#############
    PwmWork = NormPwm(KernelSeqs, True)

    return PwmWork

def KernelSeqDive(tmp_ker, seqs, Pos=True):
    """
    kernel提取每个sequence上的片段以及对应的卷积分数。
    同时保留kernel挖取的序列片段上的位置信息[序列编号，序列起始位置，结束位置]
    :param tmp_ker:
    :param seqs:
    :return:
    """
    ker_len = tmp_ker.shape[0]
    inputs = K.placeholder(seqs.shape)
    ker = K.variable(tmp_ker.reshape(ker_len, 4, 1))
    conv_result = K.conv1d(inputs, ker, padding="valid", strides=1, data_format="channels_last")
    max_idxs = K.argmax(conv_result, axis=1)
    max_Value = K.max(conv_result, axis=1)
    # sort_idxs = tensorflow.nn.top_k(tensorflow.transpose(max_Value,[1,0]), 100, sorted=True).indices

    f = K.function(inputs=[inputs], outputs=[max_idxs, max_Value])
    ret_idxs, ret = f([seqs])

    if Pos:
        seqlist = []
        SeqInfo = []
        for seq_idx in range(ret.shape[0]):
            start_idx = ret_idxs[seq_idx]
            seqlist.append(seqs[seq_idx, start_idx[0]:start_idx[0] + ker_len, :])
            SeqInfo.append([seq_idx, start_idx[0], start_idx[0] + ker_len])
        del f
        return seqlist, ret, np.asarray(SeqInfo)
    else:
        return ret


def mkdir(path):
    """
    Determine if the path exists, if it does not exist, generate this path
    :param path: Path to be generated
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return (False)



############

def runVCNNC(filePath, OutputDir, HyperParaMeters, SaveData=False):
    """
    从fasta文件直接生成最终的motif
    :param filePath: fasta文件的目录
    :param OutputDir: 所有输出结果所在的模型
    :param HyperParaMeters: 你所有需要的vCNN的超参数
    :param SaveData: 是否存储hdf5的数据结果
    :return:
    """
    
    if os.path.exists(OutputDir+"/over.txt"):
        print("already trained")
        return 0
    
    ########################生成数据集#########################
    GeneRateOneHotMatrixTest = GeneRateOneHotMatrix()
    OutputDirHdf5 = OutputDir + "/Hdf5/"
    GeneRateOneHotMatrixTest.runSimple(filePath, OutputDirHdf5, SaveData=SaveData)
    ########################训练模型#########################
    data_set = [[GeneRateOneHotMatrixTest.TrainX, GeneRateOneHotMatrixTest.TrainY],
                [GeneRateOneHotMatrixTest.TestX, GeneRateOneHotMatrixTest.TestY]]
    dataNum = GeneRateOneHotMatrixTest.TrainY.shape[0]

    input_shape = GeneRateOneHotMatrixTest.TestX[0].shape
    modelsave_output_prefix = OutputDir + "/ModleParaMeter/"
    kernel_init_dict = {str(HyperParaMeters["kernel_init_size"]): HyperParaMeters["number_of_kernel"]}
    
    auc, info, model = train_vCNN(input_shape=input_shape, modelsave_output_prefix=modelsave_output_prefix,
                                  data_set=data_set, number_of_kernel=HyperParaMeters["number_of_kernel"],
                                  init_ker_len_dict=kernel_init_dict, max_ker_len=HyperParaMeters["max_ker_len"],
                                  random_seed=HyperParaMeters["random_seed"],
                                  batch_size=HyperParaMeters["batch_size"],
                                  epoch_scheme=HyperParaMeters["epoch_scheme"])
    
    ############Select Kernel ####################
    
    DenseWeights = K.get_value(model.layers[4].kernel)
    meanValue = np.mean(np.abs(DenseWeights))
    std = np.std(np.abs(DenseWeights))
    workWeightsIndex = np.where(np.abs(DenseWeights) > meanValue-std)[0]
    kernels = recover_ker(model, "vCNN", workWeightsIndex)
    print("get kernels")
    
    PwmWorklist = []
    for ker_id in range(len(kernels)):
        kernel = kernels[ker_id]
        KernelSeqs, KSconvValue, seqinfo = KernelSeqDive(kernel, GeneRateOneHotMatrixTest.seq_pos_matrix_out,)
        KernelSeqs = np.asarray(KernelSeqs)
        PwmWork = NormPwm(KernelSeqs, True)
        PwmWorklist.append(PwmWork)
        
    #####合并前########
    
    pwm_save_dir = OutputDir + "/recover_PWM/"
    mkdir(pwm_save_dir)
    for i in range(len(PwmWorklist)):
        mkdir(pwm_save_dir + "/")
        np.savetxt(pwm_save_dir + "/" + str(i) + ".txt", PwmWorklist[i])

    del model, KernelSeqs, KSconvValue, seqinfo
    gc.collect()
    np.savetxt(OutputDir + "/over.txt", np.zeros(1))






def runCNNC(filePath, OutputDir, HyperParaMeters, SaveData=False):
    """
    从fasta文件直接生成最终的motif
    :param filePath: fasta文件的目录
    :param OutputDir: 所有输出结果所在的模型
    :param HyperParaMeters: 你所有需要的vCNN的超参数
    :param SaveData: 是否存储hdf5的数据结果
    :return:
    """
    
    if os.path.exists(OutputDir+"/over.txt"):
        print("already trained")
        return 0
    
    ########################生成数据集#########################
    GeneRateOneHotMatrixTest = GeneRateOneHotMatrix()
    OutputDirHdf5 = OutputDir + "/Hdf5/"
    GeneRateOneHotMatrixTest.runSimple(filePath, OutputDirHdf5, SaveData=SaveData)
    
    ########################训练模型#########################
    data_set = [[GeneRateOneHotMatrixTest.TrainX, GeneRateOneHotMatrixTest.TrainY],
                [GeneRateOneHotMatrixTest.TestX, GeneRateOneHotMatrixTest.TestY]]
    input_shape = GeneRateOneHotMatrixTest.TestX[0].shape
    dataNum = GeneRateOneHotMatrixTest.TrainY.shape[0]
    modelsave_output_prefix = OutputDir + "/ModleParaMeter/"
    
    auc, info, model = train_CNN(input_shape=input_shape, modelsave_output_prefix=modelsave_output_prefix,
                                 data_set=data_set, number_of_kernel=HyperParaMeters["number_of_kernel"],
                                 kernel_size=HyperParaMeters["KernelLen"], random_seed=HyperParaMeters["random_seed"],
                                 batch_size=HyperParaMeters["batch_size"], epoch_scheme=HyperParaMeters["epoch_scheme"])
    
    ############Select Kernel ####################
    DenseWeights = K.get_value(model.layers[3].kernel)
    
    meanValue = np.mean(np.abs(DenseWeights))
    std = np.std(np.abs(DenseWeights))
    workWeightsIndex = np.where(np.abs(DenseWeights) > meanValue-std)[0]
    kernels = recover_ker(model, "CNN", workWeightsIndex)
    print("get kernels")
    
    PwmWorklist = []
    
    for ker_id in range(len(kernels)):
        kernel = kernels[ker_id]
        KernelSeqs, KSconvValue,SeqInfo = KernelSeqDive(kernel, GeneRateOneHotMatrixTest.seq_pos_matrix_out)

        KernelSeqs = np.asarray(KernelSeqs)
        PwmWork = NormPwm(KernelSeqs, True)
        PwmWorklist.append(PwmWork)

    #####合并前########

    pwm_save_dir = OutputDir + "/recover_PWM/"
    mkdir(pwm_save_dir)
    for i in range(len(PwmWorklist)):
        mkdir(pwm_save_dir + "/")
        np.savetxt(pwm_save_dir + "/" + str(i) + ".txt", PwmWorklist[i])

    del model, KernelSeqs, KSconvValue,SeqInfo
    np.savetxt(OutputDir + "/over.txt", np.zeros(1))
    gc.collect()

