import os
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import h5py
import time
import math
import pickle
import pdb
from scipy.linalg import expm, sinm, cosm
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os

def DingYTransForm(KernelWeights):
    """
    根据丁阳的算法生成PWM
    :param KernelWeights:
    :return:
    """
    ExpArrayT = np.exp(KernelWeights * np.log(2.0))
    ExpArray = np.sum(ExpArrayT, axis=1, keepdims=True)
    ExpTensor = np.repeat(ExpArray, 4, axis=1)
    PWM = np.divide(ExpArrayT, ExpTensor)

    return PWM


def CalShanoyE(PWM):
    """
    计算PWM的香农熵

    :param PWM:
    :return:
    """
    Shanoylog = -np.log(PWM) / np.log(2.0)
    ShanoyE = np.sum(Shanoylog * PWM, axis=1, keepdims=True)
    return ShanoyE

def MaskWeightToMaskValue(kernelShape, MaskWeight):
    """
    由maskweight生成n*1*m的maskvalue
    :param MaskWeight:
    :return:
    """
    def sigmoid(x):
        return 1.0/(1+ np.exp(-x*1.0))

    def init_left(kernelShape,k_weights):

        k_weights_tem_2d_left = np.arange(kernelShape[0])  # shape[0]是长度
        k_weights_tem_2d_left = np.expand_dims(k_weights_tem_2d_left, 1)
        k_weights_tem_3d_left = np.repeat(k_weights_tem_2d_left, kernelShape[2], axis=1) - k_weights[0, :, :]  # shape[2]是numbers
        k_weights_3d_left = np.expand_dims(k_weights_tem_3d_left, 1)
        return k_weights_3d_left

    def init_right(kernelShape,k_weights):
        k_weights_tem_2d_right = np.arange(kernelShape[0])  # shape[0]是长度
        k_weights_tem_2d_right = np.expand_dims(k_weights_tem_2d_right, 1)
        k_weights_tem_3d_right = -(np.repeat(k_weights_tem_2d_right, kernelShape[2], axis=1)- k_weights[1, :, :])  # shape[2]是numbers
        k_weights_3d_right = np.expand_dims(k_weights_tem_3d_right, 1)
        return k_weights_3d_right

    k_weights_3d_left = init_left(kernelShape, MaskWeight)
    k_weights_3d_right = init_right(kernelShape, MaskWeight)

    k_weights_left = sigmoid(k_weights_3d_left)
    k_weights_right = sigmoid(k_weights_3d_right)
    MaskFinal = k_weights_left + k_weights_right - 1

    return MaskFinal

def ChooseTheBestModel(modelpath):
    """
    在给定特定数据集里选择表现最优的模型
    :param modelpath:
    :return: 最优模型的全部子模型
    """
    best_info_path = modelpath + "best_info.txt"
    with open(best_info_path, "r") as f:
        modelsave_output_filename = f.readlines()[0][:-1]
    modelsave_output_filename = modelpath + modelsave_output_filename.replace("Report_KernelNum-","model_KernelNum-")
    modellist = sorted(glob.glob(modelsave_output_filename.replace(".hdf5","*")), key=os.path.getmtime)
    modelReplace = glob.glob(modelsave_output_filename.replace("hdf5","checkpointer.hdf5")+"*")[0]
    modellist.remove(modelReplace)
    return modellist


def ExtractMaskKernel(modelPath):
    """
    取出模型的mask和kernel参数
    :return:
    """
    AllWeightTem = h5py.File(modelPath)
    AllWeight = AllWeightTem["model_weights"]
    l = [s for s in AllWeight.keys() if 'v_conv1d' in s]
    vCNNWeightTem = AllWeight[l[0]]
    vCNNWeight = vCNNWeightTem[vCNNWeightTem.keys()[0]]

    MaskWeight = vCNNWeight['k_weights:0'].value
    KernelWeight = vCNNWeight['kernel:0'].value

    return MaskWeight, KernelWeight


def DrawPic():
    pass


def GeneRateShannyChange(resultPath):
    """
    生成一个组模型的香农熵
    :return:
    """

    DataTypelist = glob.glob(resultPath+"/S*")
    ShannoyEtropy = {}
    MaskWeightdict = {}
    ShanoyMKValue = {}
    mode_lst = ["vCNN"]

    for DataType in DataTypelist:
        DataTypeStr = DataType.split("/")[-1]
        ShannoyEtropy[DataTypeStr] = []
        for Modeltype in mode_lst:
            Modeltype = DataType + "/" + Modeltype
            BestModellist = ChooseTheBestModel(Modeltype+"/")
            ShannoyEtropy[DataTypeStr] = []
            MaskWeightdict[DataTypeStr] = []
            ShanoyMKValue[DataTypeStr] = []
            for i in range(len(BestModellist)):
                #根据epoch选取每一个model

                modelPath = BestModellist[i]
                MaskWeight, KernelWeight = ExtractMaskKernel(modelPath)
                print("读取mask成功")
                PWM = DingYTransForm(KernelWeight)
                ShanoyE = CalShanoyE(PWM)
                MaskValue= MaskWeightToMaskValue(KernelWeight.shape, MaskWeight)
                print("生成maskvalue成功")
                ShanoylossValue = ShanoyE * MaskValue
                print("生成香农熵成功")
                ShannoyEtropy[DataTypeStr].append(ShanoyE)
                MaskWeightdict[DataTypeStr].append(np.asarray(MaskWeight)*1.0)
                ShanoyMKValue[DataTypeStr].append(ShanoylossValue)


    return ShannoyEtropy,MaskWeightdict,ShanoyMKValue

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

def draw_heat_map(data, xlabels, ylabels, names, path, vmax=None, vmin=None):
    cmap = plt.cm.get_cmap("rainbow")
    figure = plt.figure(facecolor="w")
    ax = figure.add_subplot(1, 1, 1, position=[0.1, 0.15, 0.8, 0.8])
    ax.set_title(names, fontsize=18)
    ax.set_ylabel(ylabels, fontsize=18)
    ax.set_xlabel(xlabels, fontsize=18)  # 横变成y轴，跟矩阵原始的布局情况是一样的
    if not vmax:
        vmax=data[0][0]
        vmin=data[0][0]
    for i in data:
        for j in i:
            if j>vmax:
                vmax=j
            if j<vmin:
                vmin=j
    map = ax.imshow(data, interpolation='nearest', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)
    plt.savefig(path)
    plt.savefig(path.replace(".png",".eps"))
    plt.close()

def Draw(ShannoyEtropy, MaskWeightdict, ShanoyMKValue,outputPath):
    """
    分别画香农熵的变化，maskWeight的变化，和香农熵乘以maskweight以后的变化
    :param ShannoyEtropy:
    :param MaskWeightdict:
    :param ShanoyMKValue:
    :param outputPath:
    :return:
    """
    Keylist = ShannoyEtropy.keys()

    ShannoyEtropyPath = outputPath + "/ShannoyEtropy/"
    MaskWeightPath = outputPath + "/MaskWeight/"
    ShanoyMKValuePath = outputPath + "/ShanoyMKValue/"
    mkdir(ShannoyEtropyPath)
    mkdir(MaskWeightPath)
    mkdir(ShanoyMKValuePath)

    for datakey in Keylist:

        # 画kernel的香农熵的变化情况

        SE = ShannoyEtropy[datakey]
        for i in range(len(SE)):
            SETem = SE[i]
            SETem = SETem.reshape(SETem.shape[0],SETem.shape[2])
            path = ShannoyEtropyPath + datakey + "/" + str(i) + ".png"
            mkdir(ShannoyEtropyPath + datakey)
            draw_heat_map(SETem, 'Kernel number', 'kernel length', 'Shannoy entropy',path,vmax=2,vmin=1)

        # 画mask的变化情况

        MK = MaskWeightdict[datakey]
        for i in range(len(MK)):
            MKTem = MK[i]
            MKTem = MKTem.reshape(MKTem.shape[0],MKTem.shape[2])
            path = MaskWeightPath + datakey + "/" + str(i) + ".png"
            mkdir(MaskWeightPath + datakey)
            draw_heat_map(MKTem, 'Kernel number', 'kernel length', 'Shannoy entropy',path)


        # 画mask kernel的香农loss变化情况

        SMK = ShanoyMKValue[datakey]
        for i in range(len(SMK)):
            SMKTem = SMK[i]
            SMKTem = SMKTem.reshape(SMKTem.shape[0],SMKTem.shape[2])
            path = ShanoyMKValuePath + datakey + "/" + str(i) + ".png"
            mkdir(ShanoyMKValuePath + datakey)
            draw_heat_map(SMKTem, 'Kernel number', 'kernel length', 'Shannoy entropy',path)


def Drawold(ShannoyEtropy, MaskWeightdict, ShanoyMKValue,outputPath):
    """
    分别画香农熵的变化，maskWeight的变化，和香农熵乘以maskweight以后的变化
    :param ShannoyEtropy:
    :param MaskWeightdict:
    :param ShanoyMKValue:
    :param outputPath:
    :return:
    """
    Keylist = ShannoyEtropy.keys()

    ShannoyEtropyPath = outputPath + "/ShannoyEtropy/"
    MaskWeightPath = outputPath + "/MaskWeight/"
    ShanoyMKValuePath = outputPath + "/ShanoyMKValue/"
    mkdir(ShannoyEtropyPath)
    mkdir(MaskWeightPath)
    mkdir(ShanoyMKValuePath)

    for datakey in Keylist:

        # 画kernel的香农熵的变化情况

        SE = ShannoyEtropy[datakey]
        for i in range(len(SE)):
            SETem = SE[i]
            SETem = SETem.reshape(SETem.shape[0],SETem.shape[2])
            shape = SETem.shape
            fig, ax = plt.subplots(figsize=shape)
            # 二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
            # 和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。

            sns.heatmap(SETem,
                        annot=True, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")
            # sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
            #            square=True, cmap="YlGnBu")
            ax.set_title('shannoy etropy', fontsize=18)
            ax.set_ylabel('Kernel length', fontsize=18)
            ax.set_xlabel('kernel number', fontsize=18)  # 横变成y轴，跟矩阵原始的布局情况是一样的
            plt.savefig(ShannoyEtropyPath+datakey+"_"+str(i)+".png")
            plt.close()
        # 画mask的变化情况


        MK = MaskWeightdict[datakey]
        for i in range(len(MK)):
            MKTem = MK[i]
            MKTem = MKTem.reshape(MKTem.shape[0],MKTem.shape[2])
            fig, ax = plt.subplots(figsize=MKTem.shape)
            # 二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
            # 和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
            sns.heatmap(MKTem,
                        annot=True, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")
            # sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
            #            square=True, cmap="YlGnBu")
            ax.set_title('shannoy etropy', fontsize=18)
            ax.set_ylabel('Kernel length', fontsize=18)
            ax.set_xlabel('kernel number', fontsize=18)  # 横变成y轴，跟矩阵原始的布局情况是一样的

            plt.savefig(MaskWeightPath+datakey+"_"+str(i)+".png")
            plt.close()


        # 画mask kernel的香农loss变化情况

        # SMK = ShanoyMKValue[datakey]
        # for i in range(len(SMK)):
        #     SMKTem = SMK[i]
        #     SMKTem = SMKTem.reshape(SMKTem.shape[0],SMKTem.shape[2])
        #     fig, ax = plt.subplots(figsize=(9, 9))
        #     # 二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
        #     # 和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
        #     sns.heatmap(pd.DataFrame(np.round(SMKTem, 6)),
        #                 annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")
        #     # sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
        #     #            square=True, cmap="YlGnBu")
        #     ax.set_title('shannoy etropy', fontsize=18)
        #     ax.set_ylabel('Kernel length', fontsize=18)
        #     ax.set_xlabel('kernel number', fontsize=18)  # 横变成y轴，跟矩阵原始的布局情况是一样的
        #
        #
        #     plt.savefig(ShanoyMKValuePath+datakey+".png")
        #     plt.close()

def GeneRateAvi(imgpath):

    size = (640,480)
    # 定义编码格式mpge-4
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # 定义视频文件输入对象
    video = cv2.VideoWriter(imgpath+'\saveDir.avi', fourcc, 30, size)
    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    num = len(glob.glob(imgpath+"\*.png"))
    for i in range(num):
        file_name = imgpath+ str(i) +".png"
        img = cv2.imread(file_name)
        video.write(img)
        video.write(img)
        video.write(img)
        video.write(img)
        video.write(img)

    # video.release()
    #
    # cv2.destroyAllWindows()



if __name__ == '__main__':

    outputPath = "../OutPutAnalyse/ShannoyPic/SimulationOnTwoMotif/"
    SimulationResultRoot = "../OutPutAnalyse/result/SimulationOnTwoMotif/"
    mkdir(outputPath)

    # 获取香农熵在每一轮的值
    # ShannoyEtropy, MaskWeightdict, ShanoyMKValue = GeneRateShannyChange(SimulationResultRoot)
    # 将香农熵的值转成一个个热图
    # Draw(ShannoyEtropy, MaskWeightdict, ShanoyMKValue, outputPath)

    # 将每个文件夹下的图片转成一个avi
    # import cv2
    # dirlist = glob.glob(r"C:\Users\Lijy\Desktop\files\schoolWork\schoolWork\FinalVersion\vCNNFinal\OutPutAnalyse\ShannoyPic\SimulationOnTwoMotif\ShanoyMKValue\\*")
    # for dir in dirlist:
    #     GeneRateAvi(dir+"/")

