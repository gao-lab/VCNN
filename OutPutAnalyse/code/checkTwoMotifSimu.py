# -*- coding: utf-8 -*-
'''
Analysis result
'''
import os
import pickle
import numpy as np
import glob
import h5py
import time
import math
import pickle
import pdb
# from build_models import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.switch_backend('agg')
os.environ["CUDA_VISIBLE_DEVICES"]= "0"


def get_real_data_info(data_root_path):
    return [it.split("/")[-1]+"/" for it in glob.glob(data_root_path+"*")]

# get data_info:
def get_chipseq_data_info(data_root_path):
    return [it.split("/")[-1]+"/" for it in glob.glob(data_root_path+"*")]

# record (auc and loss)
# save the history by np.array
def flat_record(rec):
    try:
        output = np.array([x for y in rec for x in y])
    except:
        output = np.array(rec)
    return output


# Traversing the path of the deepbind data set, each time calling func
# func set 3 parameter：data_root,result_root,data_info
# func return the result as a value, data_info, as a key, return a dictionary
def iter_real_path(func,data_root, result_root):
    '''
    Traversing the simu data set,
    :param func:
    :param data_root: The root directory where the data is located
    :param result_root: The root directory where the result is located
    :return:
    '''
    data_info_lst = get_real_data_info(data_root+"/*")
    ret = {}
    for data_info in data_info_lst:
        tem = func(data_root = data_root,result_root=result_root,data_info = data_info)
        if tem != None:
            ret[data_info] = tem
    return ret

def best_model_report(data_info,result_root,data_root):
    '''
     set data_into,result_root, Return a dictionary：
    key is CNN,vCNN_lg
    value is the report of each model
    :param data_info:
    :param result_root: ]
    :return:
    '''
    def get_reports(path):
        best_info_path = path+"best_info.txt"
        if os.path.isfile(best_info_path):
            with open(best_info_path, "r") as f:
                modelsave_output_filename = f.readlines()[0][:-1]
                tmp_path = modelsave_output_filename.replace("hdf5", "pkl")

                test_prediction_output = path + tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")
                with open(test_prediction_output,"r") as f:
                    ret = pickle.load(f)
                return ret
        else:
            return None
    model_lst = ["CNN","vCNN"]
    ret = {}
    pre_path = result_root + data_info
    for item in model_lst:
        ret[item] = get_reports(pre_path+item+"/")
    print(data_info)
    for mode in ret:
        tmp_dir = ret[mode]
        if tmp_dir == None:
            continue
        else:
            d = tmp_dir.tolist()
            loss = flat_record(d["loss"])
            auc = flat_record(d["auc"])
            print(mode+ "   best: loss = {0}  auc = {1}".format(loss.min(),auc.max()))
    return ret


def gen_auc_report(item, data_info,result_root,aucouttem,DatatypeAUC, BestInfo):
    """
    :param item:
    :param data_info:
    :param result_root:
    :param aucouttem:
    :param DatatypeAUC:
    :param BestInfo:
    :return:
    """

    def get_reports(path, datatype):
        rec_lst = glob.glob(path+"Report*")
        for rec in rec_lst:
            kernelnum, kernellen = extractUseInfo(rec)
            with open(rec,"r") as f:
                tmp_dir = (pickle.load(f)).tolist()
                keylist = aucouttem.keys()
                if datatype not in DatatypeAUC.keys():
                    DatatypeAUC[datatype] = tmp_dir["test_auc"]
                    BestInfo[datatype] = rec.split("/")[-1].replace("pkl", "hdf5").replace("/Report_KernelNum-", "/model_KernelNum-")
                elif DatatypeAUC[datatype] < tmp_dir["test_auc"]:
                    DatatypeAUC[datatype] = tmp_dir["test_auc"]
                    BestInfo[datatype] = rec.split("/")[-1].replace("pkl", "hdf5").replace("/Report_KernelNum-", "/model_KernelNum-")

                if datatype in keylist:
                    aucouttem[datatype].append(tmp_dir["test_auc"])
                else:
                    aucouttem[datatype]=[]
                    aucouttem[datatype].append(tmp_dir["test_auc"])

    def extractUseInfo(name):
        Knum ,KLen = name.split("/")[-1].split("_")[1:3]
        Knum = Knum.split("-")[-1]
        KLen = KLen.split("-")[-1]
        return Knum, KLen


    pre_path = result_root + data_info+"/"
    get_reports(pre_path+item+"/", data_info.replace("/",""))

    return aucouttem, DatatypeAUC, BestInfo


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

def draw_history(data_info,hist_dic,plt_type):

    def SimuTitle(data_info):
        if data_info == "TwoMotif6 ":
            return "Simulation1"
        elif data_info == "UsefulCase1 ":
            return "Simulation2"
        elif data_info == "UsefulCase2 ":
            return "Simulation3"
        elif data_info == "UsefulCase3 ":
            return "Simulation4"
        elif data_info == "UsefulCase4 ":
            return "Simulation5"
        else:
            return data_info
    save_root = "../../OutPutAnalyse/ModelAUC/MoreComplexSituation/history/"
    mkdir(save_root)
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    tmp_dic = hist_dic[data_info]

    mode_lst = ["vCNN","CNN"]
    print("ploting: "+str(plt_type)+" history:  "+data_info)
    try:
        plt.clf()
        data_info = " ".join(data_info.split("/"))
        title = SimuTitle(data_info)
        # plt.title(str(plt_type)+" history:  "+data_info)
        plt.title(str(plt_type)+" history:  "+title)

        plt.xlabel("epoch")

        plt.ylabel(str(plt_type))

        for idx,mode in enumerate(mode_lst):
            if not (plt_type == "auc" or plt_type == "loss"):
                raise ValueError("cannot support plt_type: "+str(plt_type))
            label = mode
            tmp_data = tmp_dic[mode].tolist()[plt_type]

            y = [x for it in tmp_data for x in it]
            plt.plot(np.arange(len(y)),np.array(y),label=label,color=color_list[idx]) #,label=mode,color=color_list[idx]
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)
        plt.savefig(save_root+str(plt_type)+"-"+data_info+".eps", format="eps")
    except:
        import pdb
        pdb.set_trace()

#####################################################

def load_data(dataset):
    data = h5py.File(dataset, 'r')
    sequence_code = data['sequences'].value
    label = data['labs'].value
    return ([sequence_code, label])


#####################################################
def DrawBox(dataName, path):
    """
    :param dataName:
    :param path:
    :return:
    """
    for dataInfo in dataName:
        temfilelist = glob.glob(path + "*"+dataInfo+"*.txt")
        data = []
        labels = []
        dictlist = {}
        for filename in temfilelist:
            labels.append(filename.split("/")[-1].split("_")[1])
            aa = np.loadtxt(filename)
            dictlist[labels[-1]] = list(aa)
            data.append(aa)
            print(aa.shape)
        Pddict = pd.DataFrame(dictlist)
        plt.ylim(0.5,1)
        plt.ylabel("AUC")
        plt.xlabel(dataInfo+" Data")
        # Pddict.boxplot()
        sns.boxplot(data=Pddict)
        plt.savefig(filename.replace("txt","eps"), format="eps")
        plt.savefig(filename.replace("txt","png"))
        plt.close('all')

if __name__ == '__main__':
    # auc of all models in the hyper parameter space，prove the robust
    SimulationDataRoot = "../../Data/SimulationOnTwoMotif/HDF5/"
    SimulationResultRoot = "../../OutPutAnalyse/result/SimulationOnTwoMotif/"

    # #################the best model and save the result################
    model_lst = ["vCNN", "CNN"]
    for item in model_lst:
        aucouttem={}
        DatatypeAUC = {}
        BestInfo = {}
        "best_info.txt"
        datalist = glob.glob(SimulationDataRoot+"*")
        for dataInfo in datalist:
            data_info = dataInfo.split("/")[-1]
            aucouttem, DatatypeAUC, BestInfo = gen_auc_report(item, data_info, SimulationResultRoot, aucouttem, DatatypeAUC, BestInfo)

        Key = []
        Auc = []
        for key in DatatypeAUC.keys():
            Key.append(key)
            Auc.append(DatatypeAUC[key])
            f = open(SimulationResultRoot + key+ "/"+ item + "/best_info.txt", "wb")
            f.writelines(BestInfo[key])
            f.writelines("\n")
            f.writelines("Best AUC: " + str(DatatypeAUC[key]))
            f.close()

        Df = pd.DataFrame(Auc, index=Key)
        mkdir("../../OutPutAnalyse/ModelAUC/SimulationOnTwoMotif/")
        Df.to_csv("../../OutPutAnalyse/ModelAUC/SimulationOnTwoMotif/"+ item + "AUC.csv")
        for key in aucouttem.keys():
            print(key)
            np.savetxt("../../OutPutAnalyse/ModelAUC/SimulationOnTwoMotif/" + key +"_" + item + "_auc.txt", np.asarray(aucouttem[key]))


    dataName = [
        "Simulation1",
        "Simulation2",
        "Simulation3",
    ]
    DrawBox(dataName, "../../OutPutAnalyse/ModelAUC/SimulationOnTwoMotif/")

    ###############draw history####################################################
    r = iter_real_path(best_model_report, data_root=SimulationDataRoot, result_root=SimulationResultRoot)
    for data_info in r:
        draw_history(data_info, r, "loss")
        draw_history(data_info, r, "auc")