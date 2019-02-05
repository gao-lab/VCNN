# -*- coding: utf-8 -*-
'''
vCNN_lg_core
three functions
build_CNN_model()
train_CNN(...)
train_vCNN(...)
'''
from keras.layers import Activation, Dense
from my_history import Histories
from keras.callbacks import History
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Conv1D
from vCNN_lg_core import *
from sklearn.metrics import roc_auc_score
import random
import numpy as np
import keras
import pickle
import os
import keras.backend as K
import glob
import tensorflow as tf



def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return (False)

# for CNN
def build_CNN_model(model_template, number_of_kernel, kernel_size,
                     k_pool=1,input_shape = (1000,4)):
    model_template.add(Conv1D(
        input_shape=input_shape,
        kernel_size=(kernel_size),
        filters=number_of_kernel,
        padding='same',
        strides=1))

    model_template.add(KMaxPooling(K=k_pool))
    model_template.add(keras.layers.core.Dropout(0.2))
    model_template.add(keras.layers.core.Flatten())
    model_template.add(keras.layers.core.Dense(output_dim=1))
    model_template.add(keras.layers.Activation("sigmoid"))
    sgd = keras.optimizers.RMSprop(lr=0.01)  # default 0.001
    model_template.compile(loss='binary_crossentropy', optimizer=sgd)
    return model_template


def train_CNN(input_shape,modelsave_output_prefix,data_set, number_of_kernel, kernel_size,
             random_seed, batch_size, epoch_scheme):
    '''
    Complete CNN training for a specified data set
    :param input_shape:   Sequence shape
    :param modelsave_output_prefix:
                                    the path of the model to be saved, the results of all models are saved under the path.The saved information is:：
                                    lThe model parameter with the smallest loss: ：*.checkpointer.hdf5
                                     Historical auc and loss：Report*.pkl
    :param data_set:
                                    data[[training_x,training_y],[test_x,test_y]]
    :param number_of_kernel:
                                    kernel numbers
    :param kernel_size:
                                    kernel size
    :param random_seed:
                                    random seed
    :param batch_size:
                                    batch size
    :param epoch_scheme:           training epochs
    :return:                       model auc and model name which contains hpyer-parameters


    '''

    training_set,test_set = data_set
    X_train, Y_train = training_set
    X_test, Y_test = test_set
    tf.set_random_seed(random_seed)
    random.seed(random_seed)
    model = keras.models.Sequential()
    model = build_CNN_model(model, number_of_kernel, kernel_size,
                                        k_pool=1,input_shape = input_shape)

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

        earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1)
        tmp_hist = Histories(data = [X_test,Y_test])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                      patience=50, min_lr=0.0001)
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=n_epoc, shuffle=True, validation_split=0.2,
                  verbose=2, callbacks=[checkpointer,earlystopper, reduce_lr,tmp_hist])
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


def build_vCNN(model_template, number_of_kernel, max_kernel_length, k_pool=1,input_shape=(1000,4)):
    def relu_advanced(x):
        return K.relu(x, alpha=0.5, max_value=20)
    model_template.add(VConv1D_lg(
        input_shape=input_shape,
        kernel_size=(max_kernel_length),
        filters=number_of_kernel,
        padding='same',
        strides=1))
    model_template.add(Activation(relu_advanced))
    model_template.add(keras.layers.pooling.MaxPooling1D(pool_length=10, stride=None, border_mode='valid'))
    model_template.add(KMaxPooling(K=k_pool))
    model_template.add(keras.layers.core.Dropout(0.2))
    model_template.add(keras.layers.core.Flatten())
    model_template.add(keras.layers.core.Dense(output_dim=1))
    model_template.add(keras.layers.Activation("sigmoid"))

    sgd = keras.optimizers.RMSprop(lr=0.001)  # default 0.001
    return model_template, sgd




def train_vCNN(input_shape,modelsave_output_prefix,data_set, number_of_kernel, max_ker_len,init_ker_len_dict,
             random_seed, batch_size, epoch_scheme):

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
            :param init_ker_len_dict:
                                            kernel的初始化的长度的字典集合
            :param random_seed:
                                            随机种子的数值
            :param batch_size:
                                            batch的大小
            :param epoch_scheme:
                                             epoch的数目

            :return:
                                            test set的auc，以及模型保存的位置的“模板”：[test_auc,modelsave_output_filename]
                                            replace(".hdf5", ".checkpointer.hdf5") 得到模型参数
                                            replace("/model_KernelNum-","/Report_KernelNum-") 得到保存的auc，loss的dict


            '''

    def SelectBestModel(models):
        val = [float(name.split("_")[-1].split(".c")[0]) for name in models]
        index = np.argmin(np.asarray(val))

        return models[index]
    training_set, test_set = data_set
    X_train, Y_train = training_set
    X_test, Y_test = test_set
    tf.set_random_seed(random_seed)
    random.seed(random_seed)
    model = keras.models.Sequential()
    model, sgd = build_vCNN(model, number_of_kernel, max_ker_len, input_shape=input_shape)

    model = init_mask_final(model, init_ker_len_dict,max_ker_len)
    output_path = modelsave_output_prefix
    mkdir(output_path)
    modelsave_output_filename = output_path + "/model_KernelNum-" + str(number_of_kernel) + "_initKernelLen-" + \
                                init_ker_len_dict.keys()[0]+ "_maxKernelLen-" + str(max_ker_len) + "_seed-" + str(random_seed) \
                                + "_batch_size-" + str(batch_size) + ".hdf5"
    auc_records = []
    loss_records = []

    # 模型训练
    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    tmp_hist = Histories(data = [X_test,Y_test])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                  patience=50, min_lr=0.0001)
    CrossTrain = TrainMethod()
    # 训练
    KernelWeights, MaskWeight = model.layers[0].LossKernel, model.layers[0].MaskFinal
    mu = K.cast(0.0025, dtype='float32')
    lossFunction = ShanoyLoss(KernelWeights, MaskWeight, mu=mu)
    model.compile(loss=lossFunction, optimizer=sgd, metrics=['accuracy'])
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=modelsave_output_filename.replace(".hdf5", "_2{epoch:02d}_{val_loss:.4f}.checkpointer.hdf5"), verbose=1, save_best_only=False)
    tensorboard = keras.callbacks.TensorBoard(log_dir=modelsave_output_filename.replace(".hdf5",".log"),
                                              write_images=True, batch_size=batch_size, histogram_freq=2,
                                              embeddings_data=[X_test, Y_test])

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=int(epoch_scheme), shuffle=True,
              validation_split=0.1,
              verbose=2, callbacks=[checkpointer, reduce_lr, earlystopper, tmp_hist])
    auc_records.append(tmp_hist.aucs)
    loss_records.append(tmp_hist.losses)
    # 重新load参数，计算test set auc
    modellist = glob.glob(modelsave_output_filename.replace(".hdf5", "*_2*"))
    # BestModel = np.argmax(Historys.history["val_acc"])
    modelname =SelectBestModel(modellist)
    model.load_weights(modelname)
    model.save_weights(modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"))
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
    tmp_path = modelsave_output_filename.replace("hdf5", "pkl")
    test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")
    tmp_f = open(test_prediction_output, "wb")
    pickle.dump(np.array(report_dic), tmp_f)
    tmp_f.close()
    return [test_auc, modelsave_output_filename]

