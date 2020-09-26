# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pdb
import glob
import pickle
import sys
import keras
from sklearn.metrics import roc_auc_score
sys.path.append("../../vCNN/corecode/")
from build_models import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "1"


def loadData(dataPath):
    """
    load training, validation, test
    """
    def demisionChange(X):

        X = np.squeeze(X)
        X = X.transpose(0,2,1)
        return X

    import h5py
    f = h5py.File(dataPath,"r")

    ID = f['target_labels']
    # X_train = f["train_in"].value
    # Y_train = f["train_out"].value
    X_test = f["test_in"].value
    Y_test = f["test_out"].value
    # X_val = f["valid_in"].value
    # Y_val = f["valid_out"].value
    return ID, demisionChange(X_test), Y_test
    # return demisionChange(X_train), Y_train,demisionChange(X_test), Y_test,demisionChange(X_val),Y_val


def loadoriginalResult(path):
    """

    """
    OriCsv = pd.read_csv(path)
    bassetResult = {}
    for i in range(OriCsv.shape[0]):
        bassetResult[OriCsv["Cell label"][i]] = OriCsv["Full AUC"][i]


    return bassetResult

def SelectBestModel(path):
    """

    """
    filelist = glob.glob(path+"/*pkl")
    BestModel = ""
    BestAUC = 0
    for file in filelist:
        with open(file, "r") as f:
            tmp_dir = (pickle.load(f)).tolist()
            tmpAUC = tmp_dir["test_auc"]
            if tmpAUC>BestAUC:
                BestModel = file
                BestAUC = tmpAUC
    BestModel = BestModel.replace("pkl","checkpointer.hdf5")
    BestModel = BestModel.replace("/Report_KernelNum-", "/model_KernelNum-")

    return BestModel

def bassetResultpredictAUC(modelPath, ID, TestX, TestY):
    """

    """
    vCNNresult = {}
    model = keras.models.Sequential()
    model = build_basset_model(model)
    model.load_weights(modelPath)
    PreY = model.predict(TestX)
    for i in range(len(ID)):
        test_auc = roc_auc_score(TestY[:,i], PreY[:,i])

        vCNNresult[ID[i]] = test_auc

    return vCNNresult


def vCNNpredictAUC(modelPath, ID, TestX, TestY):
    """

    """
    vCNNresult = {}
    model = keras.models.Sequential()
    model, sgd = build_vCNN(model)
    All_non_Train = []
    AllTrain = [model.layers[0].kernel, model.layers[0].bias, model.layers[0].k_weights]
    model.layers[0].trainable_weights = AllTrain
    model.layers[0].non_trainable_weights = All_non_Train
    model.load_weights(modelPath)
    PreY = model.predict(TestX)
    for i in range(len(ID)):
        test_auc = roc_auc_score(TestY[:,i], PreY[:,i])

        vCNNresult[ID[i]] = test_auc

    return vCNNresult

def Draw(bassetResult,vCNNresult):
    """

    """
    basset = []
    vCNN = []
    keylist = list(set(bassetResult.keys()).intersection(set(vCNNresult.keys())))
    for key in keylist:
        try:
            vCNN.append(vCNNresult[key])
        except:
            pdb.set_trace()
        basset.append(bassetResult[key])
    plt.scatter(basset,vCNN)
    plt.xlabel("Basset")
    plt.ylabel("vCNN-based model")
    plt.plot([0.4, 1], [0.4, 1], color='black')
    mkdir("../png/")
    print(np.average(vCNN))
    print(np.average(basset))
    plt.savefig("../png/vCNNvsBassetCNN.jpg")

def mkdir(path):
    """
    Create a directory
    :param path: Directory path
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

def main():
    """

    """
    ###读取basset的结果
    resultPath = "../result/basset/basset/"
    BestModel = SelectBestModel(resultPath)
    ID, TestX, TestY = loadData("../../data/encode_roadmap.h5")
    bassetResult = bassetResultpredictAUC(BestModel, ID, TestX, TestY)

    ###读取生成vCNN在每组数据集上的结果
    resultPath = "../result/basset/vCNN/"
    BestModel = SelectBestModel(resultPath)
    ID, TestX, TestY = loadData("../../data/encode_roadmap.h5")
    vCNNresult = vCNNpredictAUC(BestModel, ID, TestX, TestY)

    ####生成两组结果的比较图

    Draw(bassetResult, vCNNresult)




if __name__ == '__main__':
    main()
