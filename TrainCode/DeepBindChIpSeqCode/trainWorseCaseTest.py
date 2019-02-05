# -*- coding: utf-8 -*-
import time
from multiprocessing import Pool
import os
import glob
def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

# idx 模 part_mode == part_num 时候，训练该数据集
def run(tmp_cmd):
    os.system(tmp_cmd)


def run_chipseq_data(KernelLen, KernelNum, RandomSeed, WoserKeylist):
    def get_data_info_list(root_dir, WoserKeylist):
        # 11 in total
        ret = []
        for key in WoserKeylist:
            retTem = key.replace("\n","/")
            ret.append(retTem)
        return ret
    cmd = "/home/lijy/anaconda2/bin/ipython ../../corecode/main.py"
    mode="vCNN"
    data_root = "../../Data/ChIPSeqData/HDF5/"
    result_root = "../../OutPutAnalyse/result/ChIPSeq/"
    data_info_lst = get_data_info_list(data_root, WoserKeylist)

    for i in range(0, len(data_info_lst), 4):

        data_info_lstTem=data_info_lst[i:min(i+4,len(data_info_lst)-1)]
        # pool = Pool(processes=4)

        for j in range(len(data_info_lstTem)):
            data_info = data_info_lstTem[j]
            GPUNUM = int(j % 2)
            data_path = data_root + data_info
            tmp_cmd = str(cmd + " " + data_path + " " + result_root + " " + data_info + " "
                          + mode + " " + KernelLen + " " + KernelNum + " " +RandomSeed + " " +str(GPUNUM))
            print(tmp_cmd)
            time.sleep(2)
            # pool.apply_async(os.system, (tmp_cmd))
            os.system(tmp_cmd)

        # pool.close()
        # pool.join()

if __name__ == '__main__':
    file = open('./WorseKey.txt', 'r')
    WoserKeylist = file.readlines()
    ker_size_list = [24]
    number_of_ker_list = [128]
    randomSeedslist = [0, 23, 123, 345, 1234, 9, 2323, 927]
    for RandomSeed in randomSeedslist:
        for KernelNum in number_of_ker_list:
            for KernelLen in ker_size_list:
                run_chipseq_data(str(KernelLen), str(KernelNum), str(RandomSeed),WoserKeylist)