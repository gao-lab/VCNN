# -*- coding: utf-8 -*-
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')

import h5py
import numpy as np
import glob
import sys
# import keras
sys.path.append("../../corecode/")
# from build_models import *
# from sklearn.metrics import roc_curve, roc_auc_score


def loadfile(filePath):
    """

    :param filePath:
    :return:
    """
    with open(filePath, "r") as f:
        tmp_dir = (pickle.load(f)).tolist()

    losslist = tmp_dir["loss"]
    auclist = tmp_dir["auc"]

    return losslist, auclist


# def DrawCurve(auclistdict, losslistdict, OutputPath):
#     """
#
#     :param auclistdict:
#     :param losslistdict:
#     :param OutputPath:
#     :return:
#     """
#
#     fig, axs = plt.subplots(5, 10, figsize=(5 * 4, 10 * 8), squeeze=False)
#     # fig, axs = plt.subplots(3, 10, figsize=(10 * 4, 4 * 3), squeeze=False)
#
#     for i in range(len(auclistdict.keys())):
#         name = auclistdict.keys()[i]
#         Modelauclist = auclistdict[name]
#         Modellosslist = losslistdict[name]
#
#         for j in range(len(Modelauclist)):
#             auclist = np.squeeze(np.asarray(Modelauclist[j]))
#             losslist = np.squeeze(np.asarray(Modellosslist[j]))
#
#             sns.set(style="darkgrid")
#             axs[j][i].set_xlabel(name.replace("wgEncodeAwgTfbs", "")+" Seeds "+str(j), fontsize=10)
#             epochs = np.arange(len(auclist))
#             sns.lineplot(epochs, losslist,color="black",label="loss",ax = axs[j][i])
#             sns.lineplot(epochs, auclist,color="red",label="auc",ax = axs[j][i])
#             axs[j][i].set_ylim([0.5, 1.5])
#
#             axs[j][i].set_xlabel(name.replace("wgEncodeAwgTfbs", "")+" Seeds "+str(j), fontsize=10)
#
#     plt.savefig(OutputPath + "/lossAUC.eps", format="eps")
#     plt.savefig(OutputPath + "/lossAUC.png")
#     plt.close('all')

def DrawCurve(auclistdict, losslistdict, OutputPath):
    """

    :param auclistdict:
    :param losslistdict:
    :param OutputPath:
    :return:
    """

    fig, axs = plt.subplots(5, 10, figsize=(10 * 4, 5 * 8), squeeze=False)
    # fig, axs = plt.subplots(3, 10, figsize=(10 * 4, 4 * 3), squeeze=False)

    for i in range(len(auclistdict.keys())):
        name = auclistdict.keys()[i]
        Modelauclist = auclistdict[name]
        Modellosslist = losslistdict[name]

        for j in range(len(Modelauclist)):
            auclist = np.squeeze(np.asarray(Modelauclist[j]))
            losslist = np.squeeze(np.asarray(Modellosslist[j]))

            placex = int(j/10)
            placey = j - placex*10

            sns.set(style="darkgrid")
            axs[placex][placey].set_xlabel(name.replace("wgEncodeAwgTfbs", "")+" Seeds "+str(j), fontsize=10)
            epochs = np.arange(len(auclist))
            sns.lineplot(epochs, losslist,color="black",label="loss",ax = axs[placex][placey])
            sns.lineplot(epochs, auclist,color="red",label="auc",ax = axs[placex][placey])
            axs[placex][placey].set_ylim([0.5, 1.5])

            axs[placex][placey].set_xlabel(name.replace("wgEncodeAwgTfbs", "")+" Seeds "+str(j), fontsize=10)

    plt.savefig(OutputPath + "/lossAUC.eps", format="eps")
    plt.savefig(OutputPath + "/lossAUC.png")
    plt.close('all')


def DrawRoc(RocCurvedict, OutputPath):
    """

    :param RocCurvedict:
    :param OutputPath:
    :return:
    """
    fig, axs = plt.subplots(10, 3, figsize=(3 * 4, 4 * 10), squeeze=False)
    # fig, axs = plt.subplots(3, 10, figsize=(10 * 4, 4 * 3), squeeze=False)

    for i in range(len(RocCurvedict.keys())):
        name = RocCurvedict.keys()[i]
        Prelist = RocCurvedict[name]

        for j in range(len(Prelist)):
            y_true = Prelist[j][0]
            y_pre = Prelist[j][1]
            false_positive_rate, true_positive_rate, threshold = roc_curve(y_true, y_pre)
            sns.set(style="darkgrid")
            axs[j][i].set_xlabel(name.replace("wgEncodeAwgTfbs", "")+" Seeds "+str(j), fontsize=10)
            axs[j][i].plot(false_positive_rate, true_positive_rate)
            # sns.lineplot(false_positive_rate, true_positive_rate, color="black",label="loss",ax = axs[i][j])
            axs[j][i].set_xlabel(name.replace("wgEncodeAwgTfbs", "")+" Seeds "+str(j), fontsize=10)

    plt.savefig(OutputPath + "/AUCcurve.eps", format="eps")
    plt.savefig(OutputPath + "/AUCcurve.png")
    plt.close('all')



def load_data(dataset):
    """
    load training and test data set
    :param dataset: path of dataset
    :return:
    """
    data = h5py.File(dataset, 'r')
    sequence_code = data['sequences'].value
    label = data['labs'].value
    return ([sequence_code, label])


def Readpath(modelpath):
    """

    :param modelpath:
    :return:
    """
    Tem = modelpath.split("/")[-1].split("_")
    number_of_kernel = int(Tem[1].split("-")[1])
    max_ker_len = int(Tem[3].split("-")[1])
    return number_of_kernel, max_ker_len

def Modelpredict(dataset, modelpath):
    """
    输出模型阴性阳性数据集的预测结果
    :param dataset:
    :param modelpath:
    :return:
    """

    model = keras.models.Sequential()
    input_shape = dataset[0][0].shape

    number_of_kernel, max_ker_len = Readpath(modelpath)
    model, sgd = build_vCNN(model, number_of_kernel, max_ker_len, input_shape=input_shape)
    AllTrain = [model.layers[0].kernel, model.layers[0].bias, model.layers[0].k_weights]
    All_non_Train = []
    model.layers[0].trainable_weights = AllTrain
    model.layers[0].non_trainable_weights = All_non_Train
    y_true = dataset[1]
    model.load_weights(modelpath)
    y_pre = model.predict(dataset[0], batch_size=1)

    return y_true, y_pre

    # lab = dataset[1]
    # pos = np.where(lab>0)[0]
    # neg = np.where(lab==0)[0]
    #
    # PosSeq = dataset[0][pos]
    # NewSeq = dataset[0][neg]





def main():
    """

    :return:
    """
    # ReportPath = '../../OutPutAnalyse/result/ChIPSeqWorse/'
    # OutputPath = "../../OutPutAnalyse/ModelAUC/ChIPSeqWorse/AucDifference/"
    ReportPath = '../../OutPutAnalyse/result/ChIPSeq500/'
    OutputPath = "../../OutPutAnalyse/ModelAUC/ChIPSeqWorse/AucDifference/"
    DataName = ['wgEncodeAwgTfbsSydhHelas3Znf274UcdUniPk',]
                # 'wgEncodeAwgTfbsSydhHelas3Zzz3UniPk',
                # 'wgEncodeAwgTfbsSydhK562Pol3UniPk']
    deepbind_data_root = "../../Data/ChIPSeqData/HDF5/"


    auclistdict = {}
    losslistdict = {}
    RocCurvedict = {}


    for name in DataName:
        auclistdict[name] = []
        losslistdict[name] = []
        RocCurvedict[name] = []

        pathTem = ReportPath+name+"/vCNN/"

        reportlist = glob.glob(pathTem+"/Report_*")

        datasetPath = deepbind_data_root + name+"/test.hdf5"
        # dataset = load_data(datasetPath)

        for report in reportlist:

            losslist, auclist = loadfile(report)
            auclistdict[name].append(auclist)
            losslistdict[name].append(losslist)
            # modelpath = report.replace("/Report_KernelNum-","/model_KernelNum-")
            # modelpath = modelpath.replace("pkl", "checkpointer.hdf5")
            # y_true, y_pre = Modelpredict(dataset, modelpath)
            # RocCurvedict[name].append([y_true, y_pre])

    # DrawRoc(RocCurvedict, OutputPath)
    DrawCurve(auclistdict, losslistdict, OutputPath)


if __name__ == '__main__':
    main()


