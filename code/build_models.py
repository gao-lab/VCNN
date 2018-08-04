# -*- coding: utf-8 -*-
'''
support following two functions
train_CNN(...)
train_vCNN_IC(...)
train_vCNN_lg(...)
'''
from my_history import Histories
from mask_update_callback import mask_update
from keras.callbacks import LearningRateScheduler
from keras.layers import Conv1D
from core_conv_ic import Conv1D_IC
from sklearn.metrics import roc_auc_score
import random
import numpy as np
import keras
import pickle
import os
def scheduler(epoch):
	if epoch <= 20:
		return 0.01
	elif epoch%20==0:
		return 0.1/(epoch*2.0)
	else:
		return 0.1/((epoch-epoch%10+10)*2.0)


def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return (False)
# for CNN
def build_CNN_model(model_template, number_of_kernel, kernel_size,input_shape = (1000,4)):
    # kernel_init_size is int; indicates the length of the true modif inited
    # kernel_length_max indicates the max length of kernel
    model_template.add(Conv1D(
        input_shape=input_shape,
        kernel_size=(kernel_size),
        filters=number_of_kernel,
        padding='same',
        strides=1))

    model_template.add(keras.layers.GlobalMaxPooling1D())  # 0 for conv's max k pool ; 1 for kernel's max k pool
    # return(model_template)
    model_template.add(keras.layers.core.Dropout(0.2))
    model_template.add(keras.layers.core.Dense(output_dim=1))
    model_template.add(keras.layers.Activation("sigmoid"))
    sgd = keras.optimizers.RMSprop(lr=0.001)  # default 0.001

    model_template.compile(loss='binary_crossentropy', optimizer=sgd)

    return model_template
def train_CNN(input_shape,modelsave_output_prefix,data_set, number_of_kernel, kernel_size,
             random_seed, batch_size, epoch_scheme):
    '''
    完成一个指定数据集的CNN训练
    :param input_shape:
                                    序列的shape
    :param modelsave_output_prefix:
                                    保存模型的路径，所有模型的结果，都保存在该路径下.保存的信息有：
                                    loss最小的模型参数：*.checkpointer.hdf5
                                    历史的auc和loss：Report*.pkl
    :param data_set:
                                    数据集 [[training_x,training_y],[test_x,test_y]]
    :param number_of_kernel:
                                    kernel的个数
    :param kernel_size:
                                    kernel的大小
    :param random_seed:
                                    随机种子的数值
    :param batch_size:
                                    batch的大小
    :param epoch_scheme:
                                    训练的策略，一个list,每个元素是一次fit的epoch数，每次load一次最优参数
    :return:
                                    test set的auc，以及模型保存的位置的“模板”：[test_auc,modelsave_output_filename]
                                    replace(".hdf5", ".checkpointer.hdf5") 得到模型参数
                                    replace("/model_KernelNum-","/Report_KernelNum-") 得到保存的auc，loss的dict


    '''

    training_set,test_set = data_set
    X_train, Y_train = training_set
    X_test, Y_test = test_set
    np.random.seed(random_seed)
    random.seed(random_seed)
    model = keras.models.Sequential()
    model = build_CNN_model(model, number_of_kernel, kernel_size,input_shape = input_shape)

    auc_records = [] # this will record the auc of each epoch
    loss_records = [] # this will record the loss of each epoch
    output_path = modelsave_output_prefix
    mkdir(output_path)
    modelsave_output_filename = output_path + "/model_KernelNum-" + str(number_of_kernel) + "_KernelLen-" + str(
        kernel_size) + "_seed-" + str(random_seed) +"_batch_size-" + str(batch_size) + ".hdf5"

    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"),
        verbose=1, save_best_only=True)

    for n_epoc in epoch_scheme:

        earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        tmp_hist = Histories(data = [X_test,Y_test])
        change_lr = LearningRateScheduler(scheduler)

        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=n_epoc, shuffle=True, validation_split=0.05,
                  verbose=2, callbacks=[checkpointer,earlystopper, change_lr,tmp_hist])
        auc_records.append(tmp_hist.aucs)
        loss_records.append(tmp_hist.losses)
        model.load_weights(modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"))

    model.load_weights(modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"))
    y_pred = model.predict(X_test)
    test_auc = roc_auc_score(Y_test,y_pred)
    print("test_auc = {0}".format(test_auc))
    best_auc = np.array([y for x in auc_records for y in x] ).max()
    best_loss = np.array([y for x in loss_records for y in x]).min()
    print("best_auc = {0}".format(best_auc))
    print("best_loss = {0}".format(best_loss))
    report_dic = {}
    report_dic["auc"] = auc_records
    report_dic["loss"] = loss_records
    report_dic["test_auc"] = test_auc
    # save the auc and loss record
    tmp_path = modelsave_output_filename.replace("hdf5", "pkl")
    test_prediction_output = tmp_path.replace("/model_KernelNum-","/Report_KernelNum-")
    tmp_f = open(test_prediction_output,"wb")
    pickle.dump(np.array(report_dic),tmp_f)
    tmp_f.close()
    return [test_auc,modelsave_output_filename]

# for vCNN_IC

def build_vCNN_IC(model_tmp,number_of_kernel,max_ker_len,
                  init_ker_len,IC_thrd = 0.05,input_shape = (1000,4)):
    model_tmp.add(Conv1D_IC(
        input_shape=input_shape,
        kernel_init_len=init_ker_len,
        kernel_max_len = max_ker_len,
        maskTr__IC_threshold = IC_thrd,
        filters=number_of_kernel))
    model_tmp.add(keras.layers.GlobalMaxPooling1D())
    model_tmp.add(keras.layers.core.Dropout(0.2))
    model_tmp.add(keras.layers.core.Dense(output_dim=1))
    model_tmp.add(keras.layers.Activation("sigmoid"))
    sgd = keras.optimizers.RMSprop(lr=0.001)  # default 0.001
    model_tmp.compile(loss='binary_crossentropy', optimizer=sgd)
    return model_tmp

def train_vCNN_IC(input_shape,modelsave_output_prefix,data_set, number_of_kernel, max_ker_len,init_ker_len,
             random_seed, batch_size,n_epoch ,IC_thrd = 0.05,jump_trained = False):
    '''
        完成一个指定数据集的vCNN_IC训练
        :param input_shape:
                                        序列的shape
        :param modelsave_output_prefix:
                                        保存模型的路径，所有模型的结果，都保存在该路径下.保存的信息有：
                                        loss最小的模型参数：*.checkpointer.hdf5
                                        历史的auc和loss：Report*.pkl
        :param data_set:
                                        数据集 [[training_x,training_y],[test_x,test_y]]
        :param number_of_kernel:
                                        kernel的个数
        :param max_ker_len:
                                        kernel的最大的长度
        :param init_ker_len:
                                        kernel的初始化的长度
        :param random_seed:
                                        随机种子的数值
        :param batch_size:
                                        batch的大小
        :param epoch_scheme:
                                        训练的策略，一个list,第一个元素是初次训练的时间，
                                        对于vCNN_IC,从epoch_scheme第二个开始，每个元素是一次update_mask之间fit的epoch数，
                                        每次模型返回停止更新maskh后，进入下一次，load一次最优参数，reset state
        :param IC_thrd:
                                        mask更新时候的阈值
        :return:
                                        test set的auc，以及模型保存的位置的“模板”：[test_auc,modelsave_output_filename]
                                        replace(".hdf5", ".checkpointer.hdf5") 得到模型参数
                                        replace("/model_KernelNum-","/Report_KernelNum-") 得到保存的auc，loss的dict

        '''

    training_set, test_set = data_set
    X_train, Y_train = training_set
    X_test, Y_test = test_set
    np.random.seed(random_seed)
    random.seed(random_seed)
    model = keras.models.Sequential()
    model = build_vCNN_IC(model,number_of_kernel,max_ker_len,init_ker_len,IC_thrd,input_shape = input_shape)
    auc_records = []  # this will record the auc of each epoch
    loss_records = []  # this will record the loss of each epoch
    output_path = modelsave_output_prefix
    mkdir(output_path)
    modelsave_output_filename = output_path + "/model_KernelNum-" + str(number_of_kernel) + "_initKernelLen-" + \
                                str(init_ker_len) + "_maxKernelLen-" + str(max_ker_len)  +  "_seed-" + str(random_seed) \
                                + "_batch_size-" + str(batch_size) + ".hdf5"
    if jump_trained:
        tmp_path = modelsave_output_filename.replace("hdf5", "pkl")
        test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")
        if os.path.exists(test_prediction_output):
            print("!!!  trained and skipped::   ", test_prediction_output)
            return
    tmp_mask_update = mask_update(model.layers[0]) # by default, start update mask after 5epoch
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"), verbose=1, save_best_only=True)
    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    tmp_hist = Histories(data = [X_test,Y_test])
    change_lr = LearningRateScheduler(scheduler)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=n_epoch, shuffle=True, validation_split=0.05,
              verbose=2, callbacks=[checkpointer,earlystopper,tmp_hist,change_lr,tmp_mask_update])
    auc_records.append(tmp_hist.aucs)
    loss_records.append(tmp_hist.losses)


    # 重新load参数，计算test set auc
    model.load_weights(modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"))
    y_pred = model.predict(X_test)
    test_auc = roc_auc_score(Y_test, y_pred)
    print("test_auc = {0}".format(test_auc))
    best_auc = np.array([y for x in auc_records for y in x]).max()
    best_loss = np.array([y for x in loss_records for y in x]).min()
    print("best_auc = {0}".format(best_auc))
    print("best_loss = {0}".format(best_loss))
    report_dic = {}
    report_dic["auc"] = auc_records
    report_dic["loss"] = loss_records
    report_dic["test_auc"] = test_auc
    # save the auc and loss record
    tmp_path = modelsave_output_filename.replace("hdf5", "pkl")
    test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")

    tmp_f = open(test_prediction_output, "wb")
    pickle.dump(np.array(report_dic), tmp_f)
    tmp_f.close()
    return [test_auc,modelsave_output_filename]

