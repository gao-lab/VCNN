'''
compare with deepbind's baseline
'''
import glob
import pandas as pd
import pickle
from pprint import pprint as pprint
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import h5py
import csv
import os
import math


root = "../"
MIT_result_root = root + "result/MIT_result/"
MIT_sorted_root = root + "result/MIT_sorted/"

# make sure the dir is end with "/"
def check_dir_end(dir):
    dir = dir.replace("//","/")
    if not dir[-1] == "/":
        return dir +"/"
    else:
        return dir

def mkdir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return(True)
    else :
        return(False)

# get deepbind's result on each data set
def GetDeepbindResult(path = "../deepbind_pred/*"):
    Filelist = glob.glob(path)
    deepbindDict = {}
    for file_path in Filelist:
        temfile = pd.read_csv(file_path+"/metrics.txt", sep="\t")
        auc = float(temfile.ix[0,0].replace(" ","").replace("auc",""))
        filename = file_path.split("/")[-1].split("_")[0]
        deepbindDict[filename] = auc
    return deepbindDict
# load deepbind's result
def load_deepbind_baseline(root_path):
    f = open(root_path + "deepbind_baseline.pkl","r")
    return pickle.load(f)

# get a model's' best auc for each data set
def get_best_auc(model_root):
    fp_lst = glob.glob(check_dir_end(model_root)+"*.pkl")
    best_auc = -1
    for fp in fp_lst:
        f =open(fp,"r")
        d = pickle.load(f).tolist()
        tmp_auc = d["test_auc"]
        if best_auc<tmp_auc:
            best_auc = tmp_auc
    return best_auc


# get my result on auc for each data set and make it a dict
def get_MIT_auas_result(result_root,save_root):
    # get all data info
    def get_MIT_trained_data_info(result_root):
        lst = glob.glob(check_dir_end(result_root) + "*")
        print("got {0} data info altogether ".format(len(lst)))
        return [check_dir_end(it).split("/")[-2] for it in lst]
    ret = {}
    mode_lst = ["CNN","vCNN_IC"]
    data_info_lst = get_MIT_trained_data_info(result_root)
    for mode in mode_lst:
        for data_info in data_info_lst:
            if data_info not in ret:
                ret[data_info] = {}
            model_path = check_dir_end(check_dir_end(check_dir_end(result_root)+data_info) + mode)
            tmp_auc = get_best_auc(model_path)
            ret[data_info][mode] = tmp_auc
    f = open(check_dir_end(save_root)+"auas_result.pkl","w")
    pickle.dump(ret,f)
    f.close()
    return ret


# sort dataset's size
def sort_dataset():
    def load_dataset(dataset):
        data = h5py.File(dataset, 'r')
        sequence_code = data['sequences'].value
        label = data['labs'].value
        return ([sequence_code, label])
    data_root = "./MIT_data_root/Hdf5/"
    lst = glob.glob(data_root+"*")
    data_info_lst = [check_dir_end(it).split("/")[-2] for it in lst]
    ret = {}
    for data_info in data_info_lst:
        x,y = load_dataset(check_dir_end(data_root+data_info)+"train.hdf5")
        ret[data_info] = len(y)
    return ret

# check the data size, where VCNN_IC perform worse than deepbind's bench mark
def check_bad_dataset(dataset_lst):
    tot_dataset_result = sort_dataset()
    bad_lst = []
    tot_lst = [tot_dataset_result[tmp_key] for tmp_key in tot_dataset_result.keys()]
    for data_info in dataset_lst:
        bad_lst.append(tot_dataset_result[data_info])
    plt.clf()
    plt.title("Dataset analysis")
    plt.boxplot([tot_lst,bad_lst])
    plt.xticks([1,2],["total","bad performance"])
    plt.ylabel("training set size")
    plt.savefig(check_dir_end(MIT_sorted_root) + "dataset_compare.png")

# compare the result against baseline and visualize it
# in scatter plot and diff bar plot
def analysis_result():
    # get baseline result
    f = open(MIT_sorted_root + "MIT_baseline.pkl", "r")
    baseline_dic = pickle.load(f)
    auas_result_dic = get_MIT_auas_result(MIT_result_root, MIT_sorted_root)
    baseline = []
    vcnn = []
    data_info_lst = auas_result_dic.keys()
    for data_info in data_info_lst:
        a = baseline_dic[data_info]
        c = auas_result_dic[data_info]["vCNN_IC"]
        if c > 0:
            baseline.append(a)
            vcnn.append(c)

    # plot hist of diff
    diff_tot = [vcnn[idx] - baseline[idx] for idx in range(len(baseline))]
    plt.clf()
    plt.hist(diff_tot, bins=20)
    plt.title("AUC residual: VCNN - Deepbind")
    plt.ylabel("counts")
    plt.xlabel("AUC residual")
    plt.savefig(check_dir_end(MIT_sorted_root) + "AUC_residual.png")

    # plot scatter
    plt.clf()
    plt.title("AUC compare with deepbind")
    plt.ylim(0.25, 1.1)
    plt.xlim(0.25, 1.1)
    plt.plot(np.arange(100) / 100., np.arange(100) / 100., c="b")
    plt.scatter(baseline, vcnn, c="b", s=2)
    # plt.scatter(cnn,vcnn,c="r",label = "self_trained")
    plt.xlabel("deepbind AUC")
    plt.ylabel("VCNN AUC")
    plt.savefig(check_dir_end(MIT_sorted_root) + "AUC_plot.png")

# save AUC in csv files
def save_AUC():
    fs = open(os.path.join(MIT_sorted_root,"AUC.csv"), 'w')
    writer = csv.writer(fs)
    # make a dict: dataset Name: diff AUC
    f = open(MIT_sorted_root + "MIT_baseline.pkl", "r")
    diff_dict = {}
    baseline_dic = pickle.load(f)
    auas_result_dic = get_MIT_auas_result(MIT_result_root, MIT_sorted_root)
    baseline = []
    vcnn = []
    data_info_lst = auas_result_dic.keys()
    for data_info in data_info_lst:
        a = baseline_dic[data_info]
        c = auas_result_dic[data_info]["vCNN_IC"]
        if c > 0:
            baseline.append(a)
            vcnn.append(c)
            diff_dict[data_info] = c-a
    import operator
    # sort by AUC
    pre_lst = sorted(diff_dict.items(), key=operator.itemgetter(1))
    sorted_data_info_lst = [it[0] for it in pre_lst]
    sorted_data_info_lst.reverse()

    writer.writerow(["dataset name","AUC of VCNN","AUC of Deepbind"])
    for data_info in sorted_data_info_lst:
        writer.writerow([data_info,auas_result_dic[data_info]["vCNN_IC"],baseline_dic[data_info]])
    fs.close()

import sys
if len(sys.argv)<2:
    pass
elif sys.argv[1] == "visualize":
    analysis_result()
elif sys.argv[1] == "save_AUC":
    save_AUC()




