# -*- coding: utf-8 -*-
import time
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.warn("deprecated", DeprecationWarning)
    print("stop warning")
import os
import numpy as np
import sys

import glob
import pdb
from datetime import datetime
def tictoc():
    return datetime.now().minute * 60 + datetime.now().second + datetime.now().microsecond*(10**-6)
import matplotlib.pyplot as plt
sys.path.append("../")
from build_models import *
from  seq_to_matrix import *
from VCNNMDcore import *


plt.switch_backend('agg')

import subprocess, re

# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse "+line
        result.append(int(m.group("gpu_id")))
    return result

def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result

def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu


def get_session():
    """
    Select the size of the GPU memory used
    :return:
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu_lowest_memory())
    os.environ['OPENBLAS_NUM_THREADS'] = "1"
    os.environ['MKL_NUM_THREADS'] = '1'


def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False




if __name__ == '__main__':

    outputRoot = "../../result/TimeCost/"
    DataRoot = "../../chip-seqFa/"
    mkdir(outputRoot)
    get_session()
    
    HyperParaMeters={
        "kernel_init_size":12,
        "number_of_kernel":120,
        "max_ker_len":50,
        "batch_size":100,
        "epoch_scheme": 1000,
        "random_seed": 233
    }
    HyperParaMetersCNN={
        "kernel_init_size":12,
        "number_of_kernel":120,
        "max_ker_len":50,
        "KernelLen": 12,
        "batch_size":100,
        "epoch_scheme": 1000,
        "random_seed": 233
    }
    TestLenlist=[
        'wgEncodeAwgTfbsSydhHelas3Brf1UniPk',
       'wgEncodeAwgTfbsHaibA549GrPcr1xDex500pmUniPk',
       'wgEncodeAwgTfbsSydhHelas3Prdm19115IggrabUniPk',
       'wgEncodeAwgTfbsHaibHepg2Cebpdsc636V0416101UniPk',
       'wgEncodeAwgTfbsSydhK562Mafkab50322IggrabUniPk',
        "wgEncodeAwgTfbsBroadH1hescRbbp5a300109aUniPk",
        'wgEncodeAwgTfbsSydhHuvecCfosUcdUniPk']

    dataNumlist = [193, 1009, 4577, 11433, 19317, 16151, 46726]
    dataBplist = [46156, 280493, 1208141, 3038207, 5199041, 10712254, 17433246]

    TimeTest = []

    for i in range(len(TestLenlist)):
        fileName = TestLenlist[i]
        start = time.clock()
        demo_file_path = DataRoot + fileName + ".fa"
        print("dealing with: ",demo_file_path)
        runCNNC(demo_file_path, outputRoot+"/CNN/"+fileName, HyperParaMetersCNN)
        end = time.clock()
        print("timecost:",end-start)
        TimeTest.append([dataNumlist[i], dataBplist[i], end-start])
    np.savetxt("../../result/timeCostCNNB.txt", np.asarray(TimeTest))
    
    "ulimit -u 1 && python x.py"
    
    