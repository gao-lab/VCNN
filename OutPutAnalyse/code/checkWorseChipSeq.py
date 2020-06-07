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

    randomSeedslist = [16, 233, 2333, 23, 245, 34561, 3456, 4321, 12, 567,
                       2112, 8748, 7784, 5672, 3696, 8762, 1023, 121, 2030, 6944, 6558,
                       8705, 5302, 5777, 4472, 8782, 8669, 6850, 2284, 833, 5070, 3379,
                       4268, 1981, 4540, 8236, 7085, 3503, 7289, 1557, 2234, 6987, 3337,
                       7171, 7126, 9726, 920, 8957, 6098, 2451]

    ker_size_list = [24]
    number_of_ker_list = [128]
    
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
        # rec_lst = glob.glob(path+"Report*")
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


#####################################################

def load_data(dataset):
    data = h5py.File(dataset, 'r')
    sequence_code = data['sequences'].value
    label = data['labs'].value
    return ([sequence_code, label])


#####################################################
def DrawBox(dataName, path, outputName,OutputPath):
    """
    :param dataName:
    :param path:
    :return:
    """
    fig, axs = plt.subplots(4, 7, figsize=(7 * 4, 4 * 4), squeeze=False)
    NamelistOut = []

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
            dictlist[labels[-1]] = list(aa)
            data.append(aa)
            print(aa.shape)
        Pddict = pd.DataFrame(dictlist)

        # draw barplot with error bar
        r = int(i/7)
        c = i - r*7
        sns.set_style("darkgrid")

        # plt.ylabel("AUC",fontsize=1)
        axs[r][c].set_xlabel(outputtem.replace("wgEncodeAwgTfbs",""), fontsize=10)
        pvalue = stats.mannwhitneyu(Pddict["CNN"],Pddict["vCNN"], alternative="less")[1]
        if pvalue>=0.1:
            pV = pvalue
        else:
            pV = "P-value: "+ '{:.2e}'.format(pvalue)
        # axs[r][c].set_title("P-value: "+ format(pvalue,".2e"), fontsize=5)
        y, h = np.max([Pddict["CNN"].max(),Pddict["vCNN"].max()]), 0.01
        axs[r][c].set_ylim([0.7,max([y,1.1])])
        axs[r][c].set_yticks(axs[r][c].get_yticks()[:-2])

        if pvalue < 0.05:
            col = "red"
        else:
            col = "black"

        if pvalue > 0.9:
            NamelistOut.append(outputtem)

        # axs[r][c].plot([0, 0, 1, 1], [y, y + h, y + h, y], lw=1.5, c=col)
        axs[r][c].text((0 + 1) * 0.5, y + h, pV, ha='center', va='bottom', color=col)
        sns.barplot(data=Pddict, capsize=.2, ax = axs[r][c])
        

    fig.delaxes(axs[3][5])
    fig.delaxes(axs[3][6])
    plt.tight_layout()
    plt.savefig(OutputPath + "/barplot.eps", format="eps")
    plt.savefig(OutputPath + "/barplot.png")
    plt.close('all')
    return NamelistOut

def DrawErrorBar(data, path):
    """
    
    :param data:
    :param path:
    :return:
    """
    ##cal the dif
    
    vCNNresult = data["vCNN"]
    
    CNNresult = data["CNN"]
    sns.set_style("darkgrid")
    difference = {}
    Std = {"Name":[],"vCNN":[],"CNN":[]}
    
    mkdir(path)
    
    for i in range(len(CNNresult.keys())):
        key = CNNresult.keys()[i]
        difference[key] = list(np.asarray(vCNNresult[key]) - np.asarray(CNNresult[key]))
        # Std["Name"].append(key)
        Std["Name"].append(str(i))
        Std["vCNN"].append(np.std(vCNNresult[key]))
        Std["CNN"].append(np.std(CNNresult[key]))

    DFdifference = pd.DataFrame(difference)

    plt.ylabel("AUC difference", fontsize=20)
    # Pddict.boxplot()
    # sns.boxplot(data=DFdifference)
    # sns.pointplot(data=DFdifference, dodge=True, join=False,color="black", ci='sd')
    sns.barplot(data=DFdifference, capsize=.2)
    plt.savefig(path + "AUC_difference.eps", format="eps")
    plt.savefig(path + "AUC_difference.png")
    plt.close('all')
    
    StdDf = pd.DataFrame(Std)
    StdDf.to_csv(path+"std.csv")


def get_data_info():
    file = open('../../TrainCode/ChIpSeqworseData/WorseKeyCNN.txt', 'r')
    data_list = file.readlines()
    return data_list


def GetDataSize(path, Namelist):
    """
	:param path:
	:return:
	"""

    dataNumdict = {}
    dataNumlist = []

    for name in Namelist:
        data = path + name
        f = h5py.File(data + "/train.hdf5")
        num = f["sequences"].shape[0]
        dataNumdict[name] = num
        dataNumlist.append(num)
        f.close()

    return dataNumdict, dataNumlist

def GetCNNResult(name = '1layer_128motif',path = "../../OutPutAnalyse/ModelAUC/ChIPSeq/CNNResult/9_model_result.csv"):
	"""
	Seledt the best models results and output the dict
	:param path:
	:return:
	"""
	file = pd.read_csv(path)
	DictTem = file[['data_set', '1layer_128motif']]
	Dict = DictTem.set_index('data_set').T.to_dict('list')
	DictOutPut = {}

	for keys in Dict.keys():
		DictOutPut[keys] = Dict[keys][0]
	return DictOutPut





if __name__ == '__main__':
    # Analyze the auc of each hyperparameter of the model
    # and check the robustness of the model to hyperparameters
    SimulationResultRoot = "../../OutPutAnalyse/result/ChIPSeqWorse/"
    deepbind_data_root = "../../Data/ChIPSeqData/HDF5/"

    #################Analyze the model's optimal AUC and save it################
    model_lst = ["vCNN", "CNN"]
    AUCDifference = {}
    dataName = []
    for item in model_lst:
        aucouttem={}
        DatatypeAUC = {}
        BestInfo = {}
        datalist = get_data_info()
        for dataInfo in datalist:
            data_info = dataInfo.replace("\n","")
            aucouttem, DatatypeAUC, BestInfo = gen_auc_report(item, data_info, SimulationResultRoot, aucouttem, DatatypeAUC, BestInfo)
            dataName.append(data_info)
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
        mkdir("../../OutPutAnalyse/ModelAUC/ChIPSeqWorse/")
        Df.to_csv("../../OutPutAnalyse/ModelAUC/ChIPSeqWorse/"+ item + "AUC.csv")
        for key in aucouttem.keys():
            print(key)
            np.savetxt("../../OutPutAnalyse/ModelAUC/ChIPSeqWorse/" + key +"_" + item + "_auc.txt", np.asarray(aucouttem[key]))
        AUCDifference[item] = aucouttem
        
    # dataName = list(range(27))

    dataName = list(set(dataName))
    Namelist = DrawBox(dataName, "../../OutPutAnalyse/ModelAUC/ChIPSeqWorse/",dataName,"../../OutPutAnalyse/ModelAUC/ChIPSeqWorse/AucDifference/")
    # DrawErrorBar(AUCDifference, "../../OutPutAnalyse/ModelAUC/ChIPSeqWorse/AucDifference/")

    # dataNumdict, dataNumlist = GetDataSize(deepbind_data_root, Namelist)
    # print(dataNumdict)