# -*- coding: utf-8 -*-
import time
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.warn("deprecated", DeprecationWarning)
    print("stop warning")
import tensorflow as tf
from build_models import *
import sys

def mkdir(path):
    """
    create a directory
    :param path:
    :return:
    """
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return(True)
    else :
        return(False)

def load_data(dataset):
    """
    load training and test data set
    :param dataset:
    :return:
    """
    data = h5py.File(dataset, 'r')
    sequence_code = data['sequences'].value
    label = data['labs'].value
    return ([sequence_code, label])

def if_trained(path):
    """
    find if the model has been trained
    :param path: model path
    :return:
    """
    return os.path.isfile(path+"best_info.txt")


def get_session(gpu_fraction=0.5):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))




if __name__ == "__main__":

    # choose the GPU memory
    get_session(0.7)


    data_path = sys.argv[1]
    result_root = sys.argv[2]
    data_info = sys.argv[3]
    mode = sys.argv[4]
    KernelLen = int(sys.argv[5])
    KernelNum = int(sys.argv[6])
    RandomSeed = int(sys.argv[7])


    # loading the data set
    test_dataset = data_path + "test_set.hdf5"
    training_dataset = data_path + "training_set.hdf5"
    X_test, Y_test = load_data(test_dataset)
    X_train, Y_train = load_data(training_dataset)
    data_set = [[X_train,Y_train],[X_test,Y_test]]
    seq_len = X_test[0].shape[0]
    input_shape = X_test[0].shape
    print("seq shape = {0}    length = {1}".format(X_test[0].shape, seq_len))
    p_lst = [item for item in Y_test if item == 1] + [item for item in Y_train if item == 1]
    print("total_len = {0}     Positive_len = {1}".format(X_test.shape[0] + X_train.shape[0], len(p_lst)))

    # init hyper-parameter
    max_ker_len = min(int(seq_len * 0.5),40)
    batch_size = 100


    # model type
    if mode == "CNN":
        print("training CNN")
        time.sleep(2)
        result_path = result_root + data_info
        mkdir(result_path)
        modelsave_output_prefix = result_path + 'CNN/'
        mkdir(modelsave_output_prefix)

        auc, info = train_CNN(input_shape = input_shape,modelsave_output_prefix=modelsave_output_prefix,
                               data_set = data_set, number_of_kernel=KernelNum, kernel_size=KernelLen,
                               random_seed=RandomSeed, batch_size=batch_size,epoch_scheme=1000)

    elif mode == "vCNN":
        print("training vCNN")
        time.sleep(2)

        result_path = result_root + data_info
        mkdir(result_path)
        modelsave_output_prefix = result_path + 'vCNN/'
        mkdir(modelsave_output_prefix)
        kernel_init_dict = {str(KernelLen): KernelNum}

        auc, info = train_vCNN(input_shape=input_shape, modelsave_output_prefix=modelsave_output_prefix,
                               data_set=data_set, number_of_kernel=KernelNum,
                               init_ker_len_dict=kernel_init_dict, max_ker_len=max_ker_len,
                               random_seed=RandomSeed, batch_size=batch_size, epoch_scheme=1000)



