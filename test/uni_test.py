# -*- coding: utf-8 -*-
'''
test on conv_IC layer, will cover the following:
	（a）test load_weight
	（b）check mask state
	（c）check mask update separately
	（d）test if capture the wright kernel length
'''
import os
import h5py

import sys
sys.path.append("../code/")
sys.path.append("../test/")
sys.path.append("../demo/")

os.environ["CUDA_VISIBLE_DEVICES"]= "0"
from core_conv_ic import Conv1D_IC
from VCNN_utils import mkdir,load_dataset
from demo import build_demo_model
import numpy as np
import unittest
import keras
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import roc_auc_score
import pickle

test_data_path = os.path.join("../data/","simu_data/simu_02/")
test_result_path = os.path.join("../result/","test_result/")
mkdir(test_result_path)
# get the mask's head and tail
def get_edge_info(a_mask):
    a_mask = np.array(a_mask)
    assert len(a_mask.shape) == 2
    assert a_mask.shape[1] == 4
    L,_ = a_mask.shape
    tmp = a_mask.sum(axis = 1)
    lst = [idx for idx,item in enumerate(tmp) if not item == 0]
    head_len = lst[0]
    tail_len = L - lst[-1] - 1
    return [head_len,tail_len]



class Test_mask_update(unittest.TestCase):
    def setUp(self):
        self.filters = 2
        self.kernel_init_len = 10
        self.kernel_max_len = 40
        self.input_shape = [1000, 4]
        self.h_init_len = int(0.5 * (self.kernel_max_len - self.kernel_init_len))
        self.t_init_len = int(self.kernel_max_len - self.kernel_init_len - self.h_init_len)
        self.conv_layer = Conv1D_IC(input_shape=self.input_shape, filters=self.filters,
                                     kernel_init_len=self.kernel_init_len,
                                     kernel_max_len=self.kernel_max_len)

        self.conv_layer.build(self.input_shape)
        self.real_head_len_lst = [10,20]
        self.real_tail_len_lst = [20,10]
        # get a specical kernel to test mask update
        def get_certain_kerval(pre_kernel):
            ret = np.zeros_like(pre_kernel)
            for ker_idx in range(self.filters):
                tmp_ker = np.zeros_like(pre_kernel[:,:,ker_idx])
                tmp_head_len,tmp_tail_len = self.real_head_len_lst[ker_idx],\
                                            self.real_tail_len_lst[ker_idx]
                tmp_ker[tmp_head_len:-tmp_tail_len,:] = np.array([10.,1.,0.5,1.])
                ret[:,:,ker_idx] = tmp_ker
            return np.array(ret)
        tmp_ker = self.conv_layer.get_kernel()
        new_ker = get_certain_kerval(tmp_ker)
        self.conv_layer.test__set_kernel(new_ker)
    def show_updated_mask_edge(self):
        mask = self.conv_layer.get_mask()
        h_lst,t_lst = self.conv_layer.enclosed__get_mask_edge()
        tmp_mask_val = self.conv_layer.tmp_mask_val
        for ker_idx in range(self.filters):
            tmp_a_mask = mask[:,:,ker_idx]
            h_len,t_len = get_edge_info(tmp_a_mask)
            print("real: h_len = {0}, t_len = {1}".
                  format(self.real_head_len_lst[ker_idx],self.real_tail_len_lst[ker_idx]))
            print("return: h_len = {0}, t_len = {1}".format(h_len,t_len))
            print("recorded: h_len = {0}, t_len = {1}".format(h_lst[ker_idx],t_lst[ker_idx]))
            tmp_a_mask = tmp_mask_val[:, :, ker_idx]
            h_len, t_len = get_edge_info(tmp_a_mask)
            print("tmp_mask_val: h_len = {0}, t_len = {1}".format(h_len,t_len))
    def test_update_mask(self):
        if_stop = False
        while not if_stop:
            if_stop = self.conv_layer.update_mask()
        self.show_updated_mask_edge()
        print("reset state and stride~")
        self.conv_layer.reset_mask_state()
        self.conv_layer.set_mask_stride(1)
        if_stop = False
        while not if_stop:
            if_stop = self.conv_layer.update_mask()
        self.show_updated_mask_edge()


class Test_training(unittest.TestCase):
    def setUp(self,batch_size = 100):
        def load_dataset(dataset):
            data = h5py.File(dataset, 'r')
            sequence_code = data['sequences'].value
            label = data['labs'].value
            return ([sequence_code, label])
        self.filters = 128
        self.kernel_init_len = 20
        self.kernel_max_len = 40
        self.batch_size = batch_size
        self.model = keras.models.Sequential()
        self.model = build_demo_model(number_of_kernel = self.filters,
                                   max_ker_len = self.kernel_max_len,
                                   init_ker_len = self.kernel_init_len)
        self.train_dataset = load_dataset(test_data_path + "training_set.hdf5")
        self.test_dataset = load_dataset(test_data_path + "test_set.hdf5")
        self.input_shape = self.train_dataset[0].shape
        self.output_path = test_result_path
        self.modelsave_output_filename = self.output_path + "/model_KernelNum-" + str(self.filters) + "_initKernelLen-" + \
                                    str(self.kernel_init_len) + "_maxKernelLen-" + str(self.kernel_max_len)\
                                    + "_batch_size-" + str(self.batch_size)\
                                    + ".hdf5"
    def test_train_and_save(self):
        X_train, Y_train = self.train_dataset
        X_test, Y_test = self.test_dataset
        auc_records = []  # this will record the auc of each epoch
        loss_records = []  # this will record the loss of each epoch
        mkdir(self.output_path)

        # first round training
        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath=self.modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"), verbose=1, save_best_only=True)
        earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        self.model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=20, shuffle=True, validation_split=0.05,
                  verbose=2, callbacks=[checkpointer, earlystopper])

        # second round training
        for n_epoc in [20,20]:
            # load best parameter
            self.model.load_weights(self.modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"))
            # reset the state of all edges
            (self.model.layers[0]).reset_mask_state()
            if_stop = False
            print("reload weights and reset state")
            while (not if_stop):
                print("###################################################")
                print("if_stop:  ",if_stop)
                if_stop = self.model.layers[0].update_mask()
                earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
                self.model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=n_epoc, shuffle=True, validation_split=0.05,
                          verbose=2, callbacks=[checkpointer, earlystopper])
            # set a larger IC threshold
            self.model.layers[0].set_IC_threshold(0.2)
            if_stop = False

        # reload best model and calculate AUC after save
        self.model.save_weights(self.modelsave_output_filename.replace(".hdf5", ".final.hdf5"))
        self.model.load_weights(self.modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"))
        y_pred = self.model.predict(X_test)
        test_auc = roc_auc_score(Y_test, y_pred)
        print("test_auc = {0}".format(test_auc))
        report_dic = {}
        report_dic["test_auc"] = test_auc
        # save the auc and loss record
        tmp_path = self.modelsave_output_filename.replace("hdf5", "pkl")
        test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")
        tmp_f = open(test_prediction_output, "wb")
        pickle.dump(np.array(report_dic), tmp_f)
        tmp_f.close()

class Test_saving(unittest.TestCase):
    def setUp(self,batch_size = 100):
        def load_dataset(dataset):
            data = h5py.File(dataset, 'r')
            sequence_code = data['sequences'].value
            label = data['labs'].value
            return ([sequence_code, label])
        self.filters = 128
        self.kernel_init_len = 20
        self.kernel_max_len = 40
        self.batch_size = batch_size
        self.model = keras.models.Sequential()
        self.model = build_demo_model(number_of_kernel = self.filters,
                                   max_ker_len = self.kernel_max_len,
                                   init_ker_len = self.kernel_init_len)
        self.train_dataset = load_dataset(test_data_path + "training_set.hdf5")
        self.test_dataset = load_dataset(test_data_path + "test_set.hdf5")
        self.input_shape = self.train_dataset[0].shape
        self.output_path = test_result_path
        self.modelsave_output_filename = self.output_path + "/model_KernelNum-" + str(self.filters) + "_initKernelLen-" + \
                                    str(self.kernel_init_len) + "_maxKernelLen-" + str(self.kernel_max_len)\
                                    + "_batch_size-" + str(self.batch_size)\
                                    + ".hdf5"
    def test_load(self):
        X_test, Y_test = self.test_dataset
        self.model.load_weights(self.modelsave_output_filename.replace(".hdf5", ".final.hdf5"))
        y_pred = self.model.predict(X_test)
        test_auc = roc_auc_score(Y_test, y_pred)
        print("model loaded")
        print("test_auc = {0}".format(test_auc))
    def test_kernel_and_mask(self):
        self.model.load_weights(self.modelsave_output_filename.replace(".hdf5", ".final.hdf5"))
        mask = self.model.layers[0].get_mask()
        print(mask.shape)
        print(self.kernel_max_len)
        valid_lst = []
        for mask_idx in range(self.filters):
            h,t = get_edge_info(mask[:,:,mask_idx])
            tmp_len = self.kernel_max_len-h-t
            if tmp_len>4:
                valid_lst.append(tmp_len)
        print("vaild lst   ",valid_lst)
        print(len(valid_lst))





if __name__ == '__main__':
    unittest.main()
