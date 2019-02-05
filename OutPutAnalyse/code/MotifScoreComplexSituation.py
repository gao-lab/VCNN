'''
check result and visualize it
'''
import glob
import numpy as np
import math
import os
import matplotlib
from pprint import pprint
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import sys
import seaborn as sns


# help functions:

# make dir
def mkdir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return(True)
    else :
        return(False)

# make sure each dir path is end with "\"
def check_dir_last(str):
    def mkdir(path):
        isExists=os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            return(True)
        else :
            return(False)
    if str[-1] == "/":
        return str
    else:
        return str+"/"

# make a shorter form of transfac name
def clear_transfac_name(str):
    str = str.replace(".txt","").replace("Trans_V_","")
    return str

# first build class to measure the similarity between kernel and motif
# score_core is the basic class and conv_score is base on
# convolution score to measure the similarity
class score_core(object):
    def __init__(self,ker,mtf):
        self.ker = np.array(ker)
        self.mtf = np.array(mtf)
        self.ker_len,Sk = ker.shape
        self.mtf_len,Sm = mtf.shape
        assert Sk == 4
        assert Sm == 4
        self.ker = np.concatenate(
            [np.zeros_like(self.mtf),self.ker,np.zeros_like(self.mtf)],
            axis=0
        )
        self.idx = -self.mtf_len
    def __del__(self):
        del self.ker
        del self.mtf
    def call_a_score(self,seg_1,seg_2):
        # this function will motify by different son classes
        return 0
    def get_score(self):
        ret_score = -1
        ret_idx = -self.mtf_len
        for tmp_idx in range(self.ker_len+self.mtf_len):
            tmp_score = self.call_a_score(self.ker[tmp_idx:tmp_idx+self.mtf_len],self.mtf)
            assert not tmp_score<0
            if tmp_score>ret_score:
                ret_score = tmp_score
                ret_idx = tmp_idx-self.mtf_len
        return [ret_score,ret_idx]

class conv_score(score_core):
    def __init__(self,ker,mtf):
        super(conv_score,self).__init__(ker,mtf)
        return
    def call_a_score(self,seg_1,seg_2):
        seg_1 = np.array(seg_1)
        seg_2 = np.array(seg_2)
        ret = (seg_1*seg_2).sum()
        return ret

# find the best matching kernel for each motif in mtf_dic
# in the case of simulation data set, motif is given
# therefore they can be organized in a dict {mtf_name:mtf_value}
# scoring_kernels will find the most similar kernel for each motif in the dict
# based on conv_socre
# return a dict: {mtf_name:{"ker_val":,"match_idx":,"conv_score":}}
def scoring_kernels(kernels,mtf_dic):
    '''
    find the most similar kernel for each motif
    kernel has cut according to IC
    :param kernels: [kernel_len,4,kernel_number]
    :param mtf_dic: {mtf_name:mtf_value}
    :return: a dict {mtf_name:{"ker_val","match_idx","conv_score"}}
    '''
    ker_num = len(kernels)
    ret = {}
    # print(mtf_dic.keys())
    for mtf_name in mtf_dic.keys():
        mtf = mtf_dic[mtf_name]
        best_score = -1
        best_ker_idx = 0
        match_idx = 0
        for ker_idx in range(ker_num):
            ker = kernels[ker_idx]
            assert ker.shape[1] == 4
            tmp_conv = conv_score(ker,mtf)
            tmp_score,m_idx = tmp_conv.get_score()
            if tmp_score>best_score:
                best_score = tmp_score
                best_ker_idx = ker_idx
                match_idx = m_idx
        ret[mtf_name] = {}
        ret[mtf_name]["ker_val"] = kernels[best_ker_idx]
        ret[mtf_name]["match_idx"] = match_idx
        ret[mtf_name]["conv_score"] = best_score
    return ret

# give a column calculate the IC based on background distribution "non_arr"
def cal_IC(col,acc = 0.0001,non_arr = np.ones(4)*0.25):
    non_Entroy = np.array([-p * math.log(p) for p in non_arr if not p == 0.]).sum()
    col = np.array(col)
    assert not any(col<0.)
    assert abs(col.sum()-1.) < acc
    ret = non_Entroy - np.array([-p*math.log(p) for p in col if not p==0.]).sum()
    return ret

# draw conv score between kernel and motif
# draw the bar plot at the same time
# for each data set, draw the similar score of each motif in each model
# here we compared vCNN and CNN
# also save the best matching kernel's PWM as .txt file
def score_ker_mtf(ker_cut_threshold = 0.01,result_root = "simu_kerDB_root",img_save_root = "ker_mtf_score_img_dir"):
    mode_lst = ["CNN","vCNN"]
    mkdir(img_save_root)
    def cut_ker(ker):
        ker = np.array(ker)
        IC_lst = [cal_IC(it) for it in ker]
        IC_judge = [idx for idx in range(len(IC_lst)) if IC_lst[idx] > ker_cut_threshold]
        return ker[IC_judge[0]:IC_judge[-1]]
    def load_kers(root_path):
        # print(root_path)
        root_path = check_dir_last(root_path)
        p_lst = glob.glob(os.path.join(root_path,"*.txt"))
        kernels = [cut_ker(np.loadtxt(p)) for p in p_lst]
        return kernels
    def draw_box_plot(data_info,ret,score_type):
        def sort_data(lst):
            r = {}
            for it in lst:
                for key in it:
                    if key not in r:
                        r[key] = []
                    r[key].append(it[key]["conv_score"])
            return r

        save_p = check_dir_last(img_save_root) + score_type + "_" + data_info
        mode_lst = list(ret.keys())
        # print("mode_lst",mode_lst)
        mode_num = len(mode_lst)
        mtf_name_lst = []
        plot_data = {} # data stored for plot
        for idx in range(mode_num):
            score_lst = []
            mode = mode_lst[idx]
            r = sort_data(ret[mode])
            mtf_name_lst = r.keys()

            for mtf_name in mtf_name_lst:
                if mtf_name not in plot_data:
                    plot_data[mtf_name] = []
                plot_data[mtf_name].append(r[mtf_name])
        for mtf_name in plot_data.keys():
            plt.clf()
            sns.boxplot(data=plot_data[mtf_name])
            plt.ylabel("similarity score")
            plt.xticks(np.arange(2), mode_lst)
            plt.title(" ".join([data_info,mtf_name]))
            plt.savefig(save_p+"_"+mtf_name+".png")
            plt.savefig(save_p+"_"+mtf_name+".eps", format="eps")

        return
    # save the recovered kernel
    def save_best_PWM(dic,save_dir = "../ker_pwm/"):
        mkdir(save_dir)
        for data_info in dic.keys():
            for mode in dic[data_info]:
                for mtf_org_name in dic[data_info][mode]:
                    mtf_name = mtf_org_name.replace(".txt", "").replace("Trans_V_", "")
                    save_name = save_dir + mode + "-" + data_info + "-" + mtf_name + ".txt"
                    np.savetxt(save_name, dic[data_info][mode][mtf_org_name]["ker_val"])
    ret = {}
    for mode in mode_lst:
        tmp_p = check_dir_last(result_root)
        # print(tmp_p)
        tmp_lst = glob.glob(tmp_p+"*")
        data_info_lst = [check_dir_last(check_dir_last(it)).split("/")[-2] for it in tmp_lst]
        # print("data_info_lst: ",data_info_lst)
        for data_info in data_info_lst:
            mtf_dir = check_dir_last(check_dir_last(check_dir_last(mtf_root)+data_info))
            mtf_p_lst = glob.glob(mtf_dir+"*.txt")
            mtf_dic = {}
            for p in mtf_p_lst:
                mtf_name = check_dir_last(p).split("/")[-2]
                mtf_name = mtf_name.replace(".txt","").replace("simu","").replace("tot","")
                # print("mtf_name: ",mtf_name)
                mtf_dic[mtf_name] = np.loadtxt(p)
            ker_root = check_dir_last(check_dir_last(result_root + data_info ) + mode)
            # iterate through different models
            sorted_ret = []
            p_lst = glob.glob(os.path.join(ker_root,"*"))
            p_lst = [p + "/DBKer/" for p in p_lst]
            for p in p_lst:
                kernels = load_kers(p)
                r = scoring_kernels(kernels,mtf_dic)
                sorted_ret.append(r)

            if data_info not in ret.keys():
                ret[data_info] = {}
            ret[data_info][mode] = sorted_ret
    for data_info in ret:
        draw_box_plot(data_info, ret[data_info], "conv_score")
    return ret

# draw auc box plot among different hyper-parameters in each data set
def sort_auc(result_root = "simu_result_root"):
    # get all data info under data_root
    def get_data_info(data_root):
        lst = glob.glob(check_dir_last(data_root) + "*")
        # print(data_root)
        # print(lst)
        return [check_dir_last(it).split("/")[-2] for it in lst]
    # load all the auc in each data set
    # among all hyper parameters
    def get_auc_for_a_dataset(root_dir):
        ret = []
        lst = glob.glob(check_dir_last(root_dir)+"*.pkl")
        for p in lst:
            f = open(p,"r")
            d = pickle.load(f).tolist()
            ret.append(d["test_auc"])
            f.close()
        return ret
    # draw auc box-plot
    def draw_auc_boxplot(r,img_save_root = auc_box_dir):
        mkdir(img_save_root)

        def draw_a_plot(r, data_info):
            p = img_save_root + data_info + "_aucbox.png"
            plt.clf()
            plt.title(data_info + ":  auc plot")
            mode_lst = r.keys()
            data_lst = [r[it] for it in mode_lst]
            plt.boxplot(data_lst)
            plt.xticks(np.arange(len(mode_lst)) + 1, mode_lst, size='small')
            plt.ylabel("auc")
            plt.ylim(0.45, 0.9)
            plt.savefig(p)

        for data_info in r.keys():
            tmp_r = r[data_info]
            draw_a_plot(tmp_r, data_info)

    data_info_lst = get_data_info(result_root)
    # print(data_info_lst)
    mode_lst = ["vCNN_IC","CNN"]
    ret = {}
    for data_info in data_info_lst:
        for mode in mode_lst:
            result_root = result_root
            tmp_path = check_dir_last(check_dir_last(result_root+data_info)+mode)
            if data_info not in ret:
                ret[data_info] = {}
            if mode not in ret[data_info]:
                ret[data_info][mode] = []
            ret[data_info][mode] = ret[data_info][mode]+get_auc_for_a_dataset(tmp_path)
    draw_auc_boxplot(ret)
    return ret


if __name__ == "__main__":
    root = "../../"
    mtf_root = root + "/Data/MoreComplexSituation/RealMotif/"
    auc_box_dir = root + "result/auc_box/"
    simu_result_root = root + "result/simu_result/"
    simu_kerDB_root = root + "/OutPutAnalyse/MotifRebuild/MoreComplexSituation/"
    ker_mtf_score_img_dir = root + "/OutPutAnalyse/MotifScore/MoreComplexSituation/"


    score_ker_mtf(result_root = simu_kerDB_root,img_save_root = ker_mtf_score_img_dir)
