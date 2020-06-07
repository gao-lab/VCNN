# -*- coding: utf-8 -*-
'''
分析结果
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
import scipy.stats as stats


plt.switch_backend('agg')
os.environ["CUDA_VISIBLE_DEVICES"]= "0"


def get_real_data_info(data_root_path):
    return [it.split("/")[-1]+"/" for it in glob.glob(data_root_path+"*")]

# 获得chipseq数据集的data_info: 数据和结果的目录结果同上
def get_chipseq_data_info(data_root_path):
    return [it.split("/")[-1]+"/" for it in glob.glob(data_root_path+"*")]

# 记录record (auc和loss)时候，每次fit model 的结果都是一个子列表
# 调用这个函数，将结果展平，返回np.array
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
            # print(mode + "  unfinished")
            continue
        else:
            d = tmp_dir.tolist()
            loss = flat_record(d["loss"])
            auc = flat_record(d["auc"])
            print(mode+ "   best: loss = {0}  auc = {1}".format(loss.min(),auc.max()))
    return ret


def filelist(path):
    """
    Generate file names in order
    :param path:
    :return:
    """
    
    randomSeedslist = [121, 1231, 12341, 1234, 123, 432, 16, 233, 2333, 23, 245, 34561, 3456, 4321, 12, 567]
    ker_size_list = range(6, 22, 2)
    number_of_ker_list = range(64, 129, 32)
    
    name = path.split("/")[-2]
    rec_lst = []

    
    for KernelNum in number_of_ker_list:
        for KernelLen in ker_size_list:
            for random_seed in randomSeedslist:
                if name == "vCNN":
                    filename = path + "/Report_KernelNum-" + str(KernelNum) + "_initKernelLen-" + str(
                                        KernelLen)+ "_maxKernelLen-40_seed-" + str(random_seed) \
                                    + "_batch_size-100.pkl"
                else:
                    filename = path + "/Report_KernelNum-" + str(KernelNum) + "_KernelLen-" + str(
            KernelLen) + "_seed-" + str(random_seed) +"_batch_size-100.pkl"

                rec_lst.append(filename)
            
    return rec_lst
    
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
        rec_lst = filelist(path)
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

        dataNamedict = {
            "2 ":"2 motifs",
            "4 ":"4 motifs",
            "6 ":"6 motifs",
            "8 ":"8 motifs",
            "TwoDif1 ":"TwoDiffMotif1",
            "TwoDif2 ":"TwoDiffMotif2",
            "TwoDif3 ":"TwoDiffMotif3",
        }

        return dataNamedict[data_info]

    save_root = "../../OutPutAnalyse/ModelAUC/JasPerMotif/history/"
    mkdir(save_root)
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    tmp_dic = hist_dic[data_info]

    mode_lst = ["vCNN","CNN"]
    print("ploting: "+str(plt_type)+" history:  "+data_info)

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
        label = mode + "-based model"
        tmp_data = tmp_dic[mode].tolist()[plt_type]
        y = [x for it in tmp_data for x in it]
        plt.plot(np.arange(len(y)),np.array(y),label=label,color=color_list[idx]) #,label=mode,color=color_list[idx]
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
    plt.savefig(save_root+str(plt_type)+"-"+data_info+".eps", format="eps")
    plt.savefig(save_root+str(plt_type)+"-"+data_info+".png")


#####################################################

def load_data(dataset):
    data = h5py.File(dataset, 'r')
    sequence_code = data['sequences'].value
    label = data['labs'].value
    return ([sequence_code, label])


#####################################################
def DrawBox(dataName, path, outputName):
    """
    :param dataName:
    :param path:
    :return:
    """
    pvaluedict = {}
    for i in range(len(dataName)):
        dataInfo = dataName[i]
        outputtem = outputName[i]
        temfilelist = glob.glob(path +dataInfo+"*.txt")
        data = []
        labels = []
        dictlist = {}
        for filename in temfilelist:
            labels.append(filename.split("/")[-1].split("_")[1])
            aa = np.loadtxt(filename)
            dictlist[labels[-1]+"-based model"] = list(aa)
            data.append(aa)
            print(aa.shape)
        Pddict = pd.DataFrame(dictlist)
        # sns.set_style("darkgrid")
        # if i<4:
        #     plt.ylim(0.6,0.9)
        # else:
        #     plt.ylim(0.8, 1)
        plt.ylim(0.5, 1)
        plt.ylabel("AUC",fontsize=20)
        plt.xlabel(outputtem+" Data",fontsize=20)
        pvalue = stats.mannwhitneyu(Pddict["CNN-based model"],Pddict["vCNN-based model"], alternative="less")[1]

        plt.title("P-value: "+ format(pvalue,".2e"), fontsize=20)
        pvaluedict[dataInfo] = format(pvalue,".2e")
        # draw boxplot
        # Pddict.boxplot()
        # sns.boxplot(data=Pddict)
        # plt.savefig(filename.replace("txt","eps"), format="eps")
        # plt.savefig(filename.replace("txt","png"))
        # plt.close('all')
        # print(dataInfo+": ", stats.mannwhitneyu(Pddict["CNN"],Pddict["vCNN"], alternative="less")[1])

        # draw barplot with error bar

        # ax = sns.barplot(data=Pddict, capsize=.2)
        ax = sns.boxplot(data=Pddict)
        plt.setp(ax.patches, linewidth=0)
        plt.savefig(filename.replace("txt","eps"), format="eps")
        plt.savefig(filename.replace("txt","png"))
        plt.close('all')
    return pvaluedict
        
def DrawErrorBar(dataName, data, path,OutputName, pvaluedict):
    """
    
    :param data:
    :param path:
    :return:
    """
    ##cal the dif
    
    vCNNresult = data["vCNN"]
    
    CNNresult = data["CNN"]
    # sns.set_style("darkgrid")
    difference = {}
    Std = {"Name":[],"vCNN":[],"CNN":[]}
    
    
    for i in range(len(dataName)):
        key = dataName[i]
        name = OutputName[i]
        difference[name] = list(np.asarray(vCNNresult[key]) - np.asarray(CNNresult[key]))
        Std["Name"].append(key)
        Std["vCNN"].append(np.std(vCNNresult[key]))
        Std["CNN"].append(np.std(CNNresult[key]))
        
    DFdifference = pd.DataFrame(difference)
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.ylabel("AUC difference", font2)
    # Pddict.boxplot()
    # sns.boxplot(data=DFdifference)
    # sns.pointplot(data=DFdifference, dodge=True, join=False,color="black", ci='sd')
    Barplot = sns.barplot(data=DFdifference, capsize=.2)
    Barplot.set_xticklabels(
        Barplot.get_xticklabels(),
        rotation=45,
        horizontalalignment='right',
        # fontweight='light',
        fontsize=15
    )
    Pos = [0.01, 0.01, 0.03, 0.03, 0.04,0.05,0.055]
    for i in range(len(dataName)):
        name = dataName[i]
        Barplot.text(i, Pos[i],pvaluedict[name], color='black', ha="center")
    plt.ylim(-0.005, 0.06)
    plt.tight_layout()
    plt.savefig(path + "AUC_difference.eps", format="eps")
    plt.savefig(path + "AUC_difference.png")
    plt.close('all')
    
    StdDf = pd.DataFrame(Std)
    StdDf.to_csv(path+"std.csv")

if __name__ == '__main__':
    # Analyze the auc of each hyperparameter of the model
    # and check the robustness of the model to hyperparameters
    SimulationDataRoot = "../../Data/JasPerMotif/HDF5/"
    SimulationResultRoot = "../../OutPutAnalyse/result/JasPerMotif/"

    result = iter_real_path(best_model_report, data_root=SimulationDataRoot, result_root=SimulationResultRoot)
    for data_info in result:
        draw_history(data_info, result, "loss")
        draw_history(data_info, result, "auc")


    #################Analyze the model's optimal AUC and save it################
    model_lst = ["vCNN", "CNN"]
    AUCDifference = {}
    
    for item in model_lst:
        aucouttem={}
        DatatypeAUC = {}
        BestInfo = {}
        datalist = glob.glob(SimulationDataRoot+"*")
        for dataInfo in datalist:
            data_info = dataInfo.split("/")[-1]
            aucouttem, DatatypeAUC, BestInfo = gen_auc_report(item, data_info, SimulationResultRoot, aucouttem, DatatypeAUC, BestInfo)

        Key = []
        Auc = []
        for key in DatatypeAUC.keys():
            Key.append(key)
            Auc.append(DatatypeAUC[key])
            f = open(SimulationResultRoot + key + "/" + item + "/best_info.txt", "wb")
            f.writelines(BestInfo[key])
            f.writelines("\n")
            f.writelines("Best AUC: " + str(DatatypeAUC[key]))
            f.close()
        Df = pd.DataFrame(Auc, index=Key)
        mkdir("../../OutPutAnalyse/ModelAUC/JasPerMotif/")
        Df.to_csv("../../OutPutAnalyse/ModelAUC/JasPerMotif/"+ item + "AUC.csv")
        for key in aucouttem.keys():
            print(key)
            np.savetxt("../../OutPutAnalyse/ModelAUC/JasPerMotif/" + key +"_" + item + "_auc.txt", np.asarray(aucouttem[key]))
        AUCDifference[item] = aucouttem
        
    dataName = [
        "2",
        "4",
        "6",
        "8",
        "TwoDif1",
        "TwoDif2",
        "TwoDif3",
    ]

    OutputName = [
        "2 motifs",
        "4 motifs",
        "6 motifs",
        "8 motifs",
        "TwoDiffMotif1",
        "TwoDiffMotif2",
        "TwoDiffMotif3",
    ]
    pvaluedict = DrawBox(dataName, "../../OutPutAnalyse/ModelAUC/JasPerMotif/",OutputName)

    DrawErrorBar(dataName, AUCDifference, "../../OutPutAnalyse/ModelAUC/JasPerMotif/",OutputName,pvaluedict)