'''
train extra dataset
'''
import glob
import os
import pickle
import sys

root = "../"
# the place to save your data (690 datasets in deepbind's work)
MIT_data_root = ""
MIT_result_root = "../result/MIT_result/"

simu_data_root = root + "data/"
simu_result_root = root + "result/simu_result/"


def mkdir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return(True)
    else :
        return(False)

def check_dir_last(str):
    if str[-1] == "/":
        return str
    else:
        return str+"/"
def get_data_info(data_root):
    lst = glob.glob(check_dir_last(data_root)+"*")
    return [check_dir_last(it).split("/")[-2] for it in lst]

def train_MIT_dataset(data_root = MIT_data_root,result_root = MIT_result_root):
    def get_data_info():
        lst = glob.glob(data_root+"*")
        ret = [check_dir_last(it).split("/")[-2] for it in lst]
        return ret
    data_info_lst = get_data_info()
    mode_lst = ["vCNN_IC"]
    ker_len_lst = [8]
    seed_lst = [2333] # #12, 1234, 432, 3456, 123
    cmd = "THEANO_FLAGS=device=cuda,floatX=float32 " \
          "python main.py"
    mkdir(result_root)
    for seed in seed_lst:
        for data_info in data_info_lst:
            for mode in mode_lst:
                for ker_len in ker_len_lst:
                    tmp_cmd = str(cmd + " " + data_root + " " + result_root + " " + data_info + " " + mode \
                    + " " + str(seed) + " " + str(ker_len) + " " + str(1) )
                    print(tmp_cmd)
                    os.system(tmp_cmd)
                    # exit

def train_simu(data_root = simu_data_root,result_root = simu_result_root):
    data_info_lst = ["simu_03","simu_01","simu_02"]
    mode_lst = ["vCNN_IC","CNN"]
    seed_lst = [12, 1234, 432, 3456, 123]
    cmd = "THEANO_FLAGS=device=cuda,floatX=float32 " \
          "python main_simu.py"
    mkdir(result_root)
    print(data_info_lst)
    for seed in seed_lst:
        for data_info in data_info_lst:
            for mode in mode_lst:
                tmp_cmd = str(cmd + " " + data_root + " " + result_root + " " + data_info + " " + mode \
                              + " " + str(seed) + " " + str(0))
                print(tmp_cmd)
                os.system(tmp_cmd)

if len(sys.argv)<2:
    exit
else:
    train_mode = sys.argv[1]
    if train_mode == "simu":
        train_simu()
    elif train_mode == "deepbind":
        train_MIT_dataset()
    else:
        print("no matching train_mode: ",train_mode)