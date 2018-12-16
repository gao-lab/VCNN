'''
recover the kernel according to deepbind's method
'''
import h5py
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import glob
import pickle
import math
import keras.backend as K
import keras
from build_models import build_vCNN_IC,build_CNN_model
from pprint import pprint
# return a new list which is sorted by value
def sorted_by_value(dic):
    r = []
    for key, value in sorted(dic.iteritems(), key=lambda (k,v): (v,k)):
        r.append((key,value))
    return r
def mkdir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return(True)
    else :
        return(False)

# make sure the dir is end with "/"
def check_dir_end(dir):
    dir = dir.replace("//","/")
    if not dir[-1] == "/":
        return dir +"/"
    else:
        return dir

# get a list of data_info under data_root
# the data is organized like: data_root + data_info + *.hdf5
def get_simu_data_info_lst(data_root):
    lst = glob.glob(data_root + "*")
    return [check_dir_end(it).split("/")[-2] for it in lst]

# base class to recover kernel using deepbind's method:
'''
first, load the model and data set.

second, get model's kernel.

third, each kernel find it's matching sub-sequence in each sequences,
and normalize them to generate PWM.

finally, save each PWM in .txt file
'''
class kerenl_recover(object):
    def __init__(self,output_root,data_root,model_root,data_info):
        '''
        init the class
        :param output_root: to path to save result. each recovered kernel (PWM) will be saved in a .txt file.
        :param data_root: the root dir of data path
        :param model_root: the model saving path, there are many .pkl and .hdf5 files.
                            .pkl file saves the auc of model and .hdf5 file saves the model weights
                            this setting have to be satisfied during training.
        :param data_info: the data set's name. using check_dir_end(data_root) + data_info to find training data set
        '''
        self.output_root = output_root
        self.data_root = data_root
        self.model_root = model_root
        self.data_info = data_info
        self.non_IC = self.cal_Entroy(np.ones(4)*0.25)
        self.labs = None
        self.seqs = None
        self.kernel = None
        self.model = None
    def __del__(self):
        del self.seqs
        del self.labs
        del self.kernel
        del self.model
    def load_data(self,path):
        data = h5py.File(path, 'r')
        self.seqs = data['sequences'].value
    # get the best performance models' kernel
    def get_kernel(self,p):
        self.kernel = None
        self.model = None
        return
    # get the kernel path of the top5 auc, return a list of model path
    def get_top_per5_model_path(self):
        lst = glob.glob(self.model_root + "*.pkl")
        pre_dic = {}
        for fp in lst:
            f = open(fp,"r")
            d = pickle.load(f).tolist()
            tmp_auc = d["test_auc"]
            pre_dic[fp] =  tmp_auc
        sorted_list = sorted_by_value(pre_dic) # sorted by value
        re_num = int(len(sorted_list)*0.05) # the number of return list
        r = [it[0].replace(".pkl", ".checkpointer.hdf5").\
            replace("/Report_KernelNum-","/model_KernelNum-") \
            for it in sorted_list[-re_num:] ]
        return r

    # cut_edge according to IC
    def cal_Entroy(self,col):
        return np.array([-p * math.log(p) for p in col if not p == 0]).sum()
    def cut_ker(self,ker,thres = 0.2):
        idx_lst = [idx for idx,col in enumerate(ker)
                   if self.non_IC - self.cal_Entroy(col)>thres]
        return np.array(ker[idx_lst[0]:idx_lst[-1]])
    # judge if to save the kernel
    def if_save(self,ker):
        return True
    # get the kernel's info
    def get_model_info(self,p):
        return p.split("/")[-1].replace(".checkpointer.hdf5","")
    def DB_recover_kernel(self):
        self.load_data(check_dir_end(self.data_root+self.data_info)+"test_set.hdf5")
        p_lst = self.get_top_per5_model_path()
        for p in p_lst:
            self.get_kernel(p)
            model_info = self.get_model_info(p)
            print("~~~~~@@@@~~~~~~")
            print(model_info)
            print("~~~~~@@@@~~~~~~")
            for ker_idx in range(self.kernel.shape[-1]):
                tmp_ker = self.kernel[:,:,ker_idx]
                ker_len = tmp_ker.shape[0]
                count_mat = np.zeros_like(tmp_ker)
                inputs = K.placeholder(self.seqs.shape)

                ker = K.variable(tmp_ker.reshape(ker_len, 4, 1))
                conv_result = K.conv1d(inputs, ker, padding="valid", strides=1, data_format="channels_last")
                max_idxs = K.argmax(conv_result, axis=1)
                f = K.function(inputs=[inputs], outputs=[max_idxs])
                ret_idxs = f([self.seqs])[0][:, 0]
                for seq_idx, start_idx in enumerate(ret_idxs):
                    count_mat = count_mat + self.seqs[seq_idx, start_idx:start_idx + ker_len, :]
                ret = (count_mat.T / (count_mat.sum(axis=1)).T).T
                del f
                ret = self.cut_ker(ret)
                if self.if_save(ret):
                    mkdir(os.path.join(self.output_root,model_info))
                    sp = os.path.join(self.output_root, \
                    model_info,str(ker_idx)+".txt")
                    np.savetxt(sp,ret)
            # delete the used model
            del self.model
            del self.kernel
        return
# using deepbind's method recover VCNN_IC's kernel


# requirement:
'''
(1) under model_root dir, there are series of model_name + ".pkl"
and model_name + ."checkpointer.hdf5" files

(2) the file name have to be organized to indicate model's hyper-parameters,
thus model_name can be decoded as following:

    # decode model's parameter from its name
            for it in model_name.split("_"):
                tmp = it.split("-")
                if tmp[0] == "initKernelLen":
                    initKernelLen = int(tmp[1])
                elif tmp[0] == "KernelNum":
                    KernelNum = int(tmp[1])
                elif tmp[0] == "maxKernelLen":
                    maxKernelLen = int(tmp[1])
(3) .pkl file saves a python dict and can be loaded in this way:
            # fp is the full path a .pkl file
            # is has a key: "test_auc", which value is model's auc on test set.
            f = open(fp,"r")
            d = pickle.load(f).tolist()
            tmp_auc = d["test_auc"]
'''

# usage:
'''
# init the class first
auas_vCNN = vCNN_IC_kernel_recover(output_root,data_root,model_root,data_info)
# using DB_recover_kernel method to generate PWM and save it in output_root
auas_vCNN.DB_recover_kernel()
'''

class vCNN_IC_kernel_recover(kerenl_recover):
    '''
    using deepbind's method recover VCNN_IC's kernel
    '''
    def __init__(self,output_root,data_root,model_root,data_info):
        '''
        init the class, same as kernel_recover class
        :param output_root: to path to save result. each recovered kernel (PWM) will be saved in a .txt file.
        :param data_root: the root dir of data path
        :param model_root: the model saving path, there are many .pkl and .hdf5 files.
                            .pkl file saves the auc of model and .hdf5 file saves the model weights
                            this setting have to be satisfied during training.
        :param data_info: the data set's name. using check_dir_end(data_root) + data_info to find training data set
        '''
        super(vCNN_IC_kernel_recover,self).__init__(output_root,data_root,model_root,data_info)
        return
    def if_save(self, ker) :
        if not len(ker)>4:
            return False
        else:
            return True

    def get_kernel(self,p):
        def get_model_parater(p):
            initKernelLen = -1
            KernelNum = -1
            maxKernelLen = -1
            model_name = p.split("/")[-1].split(".")[0]
            for it in model_name.split("_"):
                tmp = it.split("-")
                if tmp[0] == "initKernelLen":
                    initKernelLen = int(tmp[1])
                elif tmp[0] == "KernelNum":
                    KernelNum = int(tmp[1])
                elif tmp[0] == "maxKernelLen":
                    maxKernelLen = int(tmp[1])
            return [initKernelLen,KernelNum,maxKernelLen]
        def show_mask_len(mask):
            mask__head_len_lst = []
            mask__tail_len_lst = []
            for ker_id in range(mask.shape[-1]):
                tmp_mask_slide = mask[:,1,ker_id]
                tmp_valid_idx = [idx for idx,item in enumerate(tmp_mask_slide) if item == 1]
                tmp_head_len = tmp_valid_idx[0]
                tmp_tail_len = 50 - tmp_valid_idx[-1] - 1
                mask__head_len_lst.append(tmp_head_len)
                mask__tail_len_lst.append(tmp_tail_len)
            valid_len = [50 - mask__head_len_lst[idx] - mask__tail_len_lst[idx] for idx in range(mask.shape[-1])]
            rep_dic = {}
            for it in valid_len:
                if it not in rep_dic:
                    rep_dic[it] = 0
                rep_dic[it] = rep_dic[it] + 1
            pprint(rep_dic)
        initKernelLen, KernelNum, maxKernelLen = get_model_parater(p)
        self.model = keras.models.Sequential()
        self.model = build_vCNN_IC(model_tmp=self.model,
                                   number_of_kernel= KernelNum,
                                   max_ker_len = maxKernelLen,
                                   init_ker_len = initKernelLen)
        self.model.load_weights(p.replace(".pkl",".checkpointer.hdf5").replace("/Report_KernelNum-","/model_KernelNum-"))
        self.model.layers[0].mask__head_len_lst = None
        mask = self.model.layers[0].get_mask()
        # show_mask_len(mask)
        self.kernel = self.model.layers[0].get_kernel()*mask
        self.model.layers[0].enclosed__show_mask_info()
        return

    def show_kerLen(self):
        print("in model root: ",self.model_root)
        lst = glob.glob(self.model_root + "*.pkl")
        for p in lst:
            self.get_kernel(p)
        return


#
class CNN_kernel_recover(kerenl_recover):
    '''
    using deepbind's method recover VCNN_IC's kernel
    '''
    def __init__(self,output_root,data_root,model_root,data_info):
        '''
        init the class, same as kernel_recover class
        :param output_root: to path to save result. each recovered kernel (PWM) will be saved in a .txt file.
        :param data_root: the root dir of data path
        :param model_root: the model saving path, there are many .pkl and .hdf5 files.
                            .pkl file saves the auc of model and .hdf5 file saves the model weights
                            this setting have to be satisfied during training.
        :param data_info: the data set's name. using check_dir_end(data_root) + data_info to find training data set
        '''
        super(CNN_kernel_recover,self).__init__(output_root,data_root,model_root,data_info)
        return
    def if_save(self, ker) :
        if not len(ker)>4:
            return False
        else:
            return True

    def get_kernel(self,p):
        def get_model_parater(p):
            KernelNum = -1
            KernelLen = -1
            model_name = p.split("/")[-1].split(".")[0]
            for it in model_name.split("_"):
                tmp = it.split("-")
                if tmp[0] == "KernelLen":
                    KernelLen = int(tmp[1])
                elif tmp[0] == "KernelNum":
                    KernelNum = int(tmp[1])
            print(p,[KernelLen,KernelNum])
            return [KernelLen,KernelNum]
        def show_mask_len(mask):
            mask__head_len_lst = []
            mask__tail_len_lst = []
            for ker_id in range(mask.shape[-1]):
                tmp_mask_slide = mask[:,1,ker_id]
                tmp_valid_idx = [idx for idx,item in enumerate(tmp_mask_slide) if item == 1]
                tmp_head_len = tmp_valid_idx[0]
                tmp_tail_len = 50 - tmp_valid_idx[-1] - 1
                mask__head_len_lst.append(tmp_head_len)
                mask__tail_len_lst.append(tmp_tail_len)
            valid_len = [50 - mask__head_len_lst[idx] - mask__tail_len_lst[idx] for idx in range(mask.shape[-1])]
            rep_dic = {}
            for it in valid_len:
                if it not in rep_dic:
                    rep_dic[it] = 0
                rep_dic[it] = rep_dic[it] + 1
            pprint(rep_dic)
        KernelLen, KernelNum = get_model_parater(p)
        self.model = keras.models.Sequential()
        self.model = build_CNN_model(model_template=self.model,
                                     number_of_kernel= KernelNum,
                                     kernel_size = KernelLen)
        self.model.load_weights(p.replace(".pkl",".checkpointer.hdf5").replace("/Report_KernelNum-","/model_KernelNum-"))
        self.kernel = self.model.layers[0].get_weights()[0]
        return


    def show_kerLen(self):
        print("in model root: ",self.model_root)
        lst = glob.glob(self.model_root + "*.pkl")
        for p in lst:
            self.get_kernel(p)
        return
# the demo below is used to recover kernel in NIPS submission
def demo(data_info):
    root = "../"
    simu_data_dir = root + "data/"
    simu_model_result = root + "result/simu_result/"
    simu_ker_root = root + "result/simu_kerDB/"
    print(data_info)
    output_root = check_dir_end(check_dir_end(simu_ker_root) + data_info)
    vCNN_IC_model_root = check_dir_end(check_dir_end(check_dir_end(simu_model_result) + data_info) + "vCNN_IC/")
    CNN_model_root = check_dir_end(check_dir_end(check_dir_end(simu_model_result) + data_info) + "CNN/")
    data_root = simu_data_dir
    mkdir(output_root)
    vCNN_IC_output_root = output_root + "vCNN_IC/"
    CNN_output_root = output_root + "CNN/"
    mkdir(vCNN_IC_output_root)
    mkdir(CNN_output_root)
    auas_vCNN = vCNN_IC_kernel_recover(vCNN_IC_output_root, data_root, vCNN_IC_model_root, data_info)

    auas_CNN = CNN_kernel_recover(CNN_output_root, data_root, CNN_model_root, data_info)
    auas_CNN.DB_recover_kernel()
    auas_vCNN.DB_recover_kernel()
    del auas_CNN
    del auas_vCNN

import sys
if len(sys.argv)<3:
    pass
elif sys.argv[1]=="simu_recover":
    demo(sys.argv[2])
