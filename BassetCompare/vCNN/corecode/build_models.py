# -*- coding: utf-8 -*-
'''
Build models and trainging scripyt
We provide three different functions for training different models.
Where train_CNN is used to train CNN, train_vCNN is used to train vCNN, and train_vCNNSEL is used to specifically train vCNN and save the results of each step to show the change of Shannon entropy.
train_CNN(...)
train_vCNN(...)
train_vCNNSEL(...)
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
    """
    Determine if the path exists, if it does not exist, generate this path
    :param path: Path to be generated
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return (False)

# for CNN
def build_basset_model(model, input_shape = (600,4)):
    """
    build basset original model
    :param model:
    :param rho:
    :param epsilon:
    :param input_shape:
    :return:
    """
    model.add(Conv1D(
        input_shape=input_shape,
        kernel_size=(19),
        filters=300,
        padding='same',
        strides=1))


    model.add(keras.layers.BatchNormalization())
    model.add(Activation("relu"))
    model.add(keras.layers.pooling.MaxPooling1D(pool_length=3, stride=None, border_mode='valid'))

    model.add(Conv1D(
        input_shape=input_shape,
        kernel_size=(11),
        filters=200,
        padding='same',
        strides=1))
    model.add(keras.layers.BatchNormalization())
    model.add(Activation("relu"))
    model.add(keras.layers.pooling.MaxPooling1D(pool_length=4, stride=None, border_mode='valid'))

    model.add(Conv1D(
        input_shape=input_shape,
        kernel_size=(7),
        filters=200,
        padding='same',
        strides=1))
    model.add(keras.layers.BatchNormalization())
    model.add(Activation("relu"))
    model.add(keras.layers.pooling.MaxPooling1D(pool_length=4, stride=None, border_mode='valid'))
    model.add(keras.layers.core.Flatten())

    model.add(keras.layers.core.Dense(output_dim=1000))
    model.add(Activation("relu"))
    model.add(keras.layers.core.Dropout(0.3))

    model.add(keras.layers.core.Dense(output_dim=1000))
    model.add(Activation("relu"))
    model.add(keras.layers.core.Dropout(0.3))

    model.add(keras.layers.core.Dense(output_dim=164))
    model.add(keras.layers.Activation("sigmoid"))

    sgd = keras.optimizers.RMSprop(lr=0.002, rho=0.98)

    model.compile(loss='binary_crossentropy', optimizer=sgd)
    return model


def train_basset(modelsave_output_prefix,dataPath,random_seed,
                 batch_size, epoch_scheme):
    """

    """

    auc_records = [] # this will record the auc of each epoch
    loss_records = [] # this will record the loss of each epoch
    output_path = modelsave_output_prefix
    mkdir(output_path)
    modelsave_output_filename = modelsave_output_prefix + "/model_seed-" + str(random_seed) + ".hdf5"

    tmp_path = modelsave_output_filename.replace("hdf5", "pkl")
    test_prediction_output = tmp_path.replace("/model_", "/Report_")
    if os.path.exists(test_prediction_output):
        print("already Trained")
        print(test_prediction_output)
        return 0,0
    X_train, Y_train,X_test, Y_test,X_val,Y_val = loadData(dataPath)

    tf.set_random_seed(random_seed)
    random.seed(random_seed)
    model = keras.models.Sequential()
    model = build_basset_model(model)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),patience=20, min_lr=0.0001)
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"),
        verbose=1, save_best_only=True)
    
    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    tmp_hist = Histories(data = [X_test,Y_test])
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epoch_scheme, shuffle=True,
              validation_data=(X_val,Y_val),
              verbose=2, callbacks=[checkpointer,earlystopper, reduce_lr, tmp_hist])
    auc_records.append(tmp_hist.aucs)
    loss_records.append(tmp_hist.losses)
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
    test_prediction_output = tmp_path.replace("/model_","/Report_")
    tmp_f = open(test_prediction_output,"wb")
    pickle.dump(np.array(report_dic),tmp_f)
    tmp_f.close()
    return [test_auc,modelsave_output_filename]


def build_vCNN(model,input_shape=(600,4)):
    """

    Building a vCNN model
    :param model: Input model
    :param number_of_kernel:number of kernel
    :param kernel_size: kernel size
    :param k_pool: Former K  maxpooling
    :param input_shape: Sequence shape
    :return:
    """


    model.add(VConv1D_lg(
        input_shape=input_shape, kernel_size=40, filters=300,
        padding='same', strides=1))

    model.add(keras.layers.BatchNormalization())
    model.add(Activation("relu"))
    model.add(keras.layers.pooling.MaxPooling1D(pool_length=3, stride=None, border_mode='valid'))

    model.add(Conv1D(
        input_shape=input_shape,
        kernel_size=(11),
        filters=200,
        padding='same',
        strides=1))
    model.add(keras.layers.BatchNormalization())
    model.add(Activation("relu"))
    model.add(keras.layers.pooling.MaxPooling1D(pool_length=4, stride=None, border_mode='valid'))

    model.add(Conv1D(
        input_shape=input_shape,
        kernel_size=(7),
        filters=200,
        padding='same',
        strides=1))
    model.add(keras.layers.BatchNormalization())
    model.add(Activation("relu"))
    model.add(keras.layers.pooling.MaxPooling1D(pool_length=4, stride=None, border_mode='valid'))
    model.add(keras.layers.core.Flatten())

    model.add(keras.layers.core.Dense(output_dim=1000))
    model.add(Activation("relu"))
    model.add(keras.layers.core.Dropout(0.3))

    model.add(keras.layers.core.Dense(output_dim=1000))
    model.add(Activation("relu"))
    model.add(keras.layers.core.Dropout(0.3))

    model.add(keras.layers.core.Dense(output_dim=164))
    model.add(keras.layers.Activation("sigmoid"))

    sgd = keras.optimizers.RMSprop(lr=0.002, rho=0.98)

    model.compile(loss='binary_crossentropy', optimizer=sgd)

    return model, sgd


def loadData(dataPath):
    """
    load training, validation, test
    """
    def demisionChange(X):

        X = np.squeeze(X)
        X = X.transpose(0,2,1)
        return X

    import h5py
    f = h5py.File(dataPath,"r")

    # ID = np.where(f['target_labels'].value == data_info)[0][0]
    X_train = f["train_in"].value
    Y_train = f["train_out"].value
    X_test = f["test_in"].value
    Y_test = f["test_out"].value
    X_val = f["valid_in"].value
    Y_val = f["valid_out"].value

    return demisionChange(X_train), Y_train,demisionChange(X_test), Y_test,demisionChange(X_val),Y_val




def train_vCNN(modelsave_output_prefix,dataPath,
             random_seed, batch_size, epoch_scheme):

    '''
    Complete vCNN training for a specified data set, only save the best model
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



    layerlist = [0]
    kernelsizelist = [19]

    mkdir(modelsave_output_prefix)
    modelsave_output_filename = modelsave_output_prefix + "/model_seed-" + str(random_seed)+ ".hdf5"

    tmp_path = modelsave_output_filename.replace("hdf5", "pkl")
    test_prediction_output = tmp_path.replace("/model_", "/Report_")
    if os.path.exists(test_prediction_output):
        print("already Trained")
        print(test_prediction_output)
        return 0,0
    auc_records = []
    loss_records = []

    X_train, Y_train,X_test, Y_test,X_val,Y_val = loadData(dataPath)
    tf.set_random_seed(random_seed)
    random.seed(random_seed)
    model = keras.models.Sequential()
    model, sgd = build_vCNN(model)
    model = init_mask_final(model, layerlist, kernelsizelist, 40)

    # 模型训练
    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    tmp_hist = Histories(data = [X_val,Y_val])
    CrossTrain = TrainMethod()
    # 训练
    KernelWeights, MaskWeight = model.layers[0].LossKernel, model.layers[0].MaskFinal
    mu = K.cast(0.0025, dtype='float32')
    lossFunction = ShanoyLoss(KernelWeights, MaskWeight, mu=mu)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),patience=20, min_lr=0.0001)
    model.compile(loss=lossFunction, optimizer=sgd, metrics=['accuracy'])
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"), verbose=1, save_best_only=True)

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=int(epoch_scheme), shuffle=True,
              validation_data=(X_val,Y_val),
              verbose=2, callbacks=[checkpointer, earlystopper, reduce_lr, tmp_hist, CrossTrain])
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

    tmp_f = open(test_prediction_output, "wb")
    pickle.dump(np.array(report_dic), tmp_f)
    tmp_f.close()
    return [test_auc, modelsave_output_filename]


def build_3vCNN(model, input_shape=(600, 4)):
    """

    Building a vCNN model
    :param model: Input model
    :param number_of_kernel:number of kernel
    :param kernel_size: kernel size
    :param k_pool: Former K  maxpooling
    :param input_shape: Sequence shape
    :return:
    """

    model.add(VConv1D_lg(
        input_shape=input_shape, kernel_size=40, filters=300,
        padding='same', strides=1))

    model.add(keras.layers.BatchNormalization())
    model.add(Activation("relu"))
    model.add(keras.layers.pooling.MaxPooling1D(pool_length=3, stride=None, border_mode='valid'))

    model.add(VConv1D_lg(
        input_shape=input_shape, kernel_size=40, filters=200,
        padding='same', strides=1))

    model.add(keras.layers.BatchNormalization())
    model.add(Activation("relu"))
    model.add(keras.layers.pooling.MaxPooling1D(pool_length=4, stride=None, border_mode='valid'))

    model.add(VConv1D_lg(
        input_shape=input_shape, kernel_size=40, filters=200,
        padding='same', strides=1))

    model.add(keras.layers.BatchNormalization())
    model.add(Activation("relu"))
    model.add(keras.layers.pooling.MaxPooling1D(pool_length=4, stride=None, border_mode='valid'))
    model.add(keras.layers.core.Flatten())

    model.add(keras.layers.core.Dense(output_dim=1000))
    model.add(Activation("relu"))
    model.add(keras.layers.core.Dropout(0.3))

    model.add(keras.layers.core.Dense(output_dim=1000))
    model.add(Activation("relu"))
    model.add(keras.layers.core.Dropout(0.3))

    model.add(keras.layers.core.Dense(output_dim=164))
    model.add(keras.layers.Activation("sigmoid"))

    sgd = keras.optimizers.RMSprop(lr=0.002, rho=0.98)

    model.compile(loss='binary_crossentropy', optimizer=sgd)

    return model, sgd


def train_3vCNN(modelsave_output_prefix, dataPath,random_seed, batch_size, epoch_scheme):
    '''
    Complete vCNN training for a specified data set, only save the best model
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

    layerlist = [0,4,8]
    kernelsizelist = [19,11,7]

    mkdir(modelsave_output_prefix)
    modelsave_output_filename = modelsave_output_prefix + "/model_seed-" + str(random_seed) + ".hdf5"

    tmp_path = modelsave_output_filename.replace("hdf5", "pkl")
    test_prediction_output = tmp_path.replace("/model_", "/Report_")
    if os.path.exists(test_prediction_output):
        print("already Trained")
        print(test_prediction_output)
        return 0, 0
    auc_records = []
    loss_records = []

    X_train, Y_train,X_test, Y_test,X_val,Y_val = loadData(dataPath)
    tf.set_random_seed(random_seed)
    random.seed(random_seed)
    model = keras.models.Sequential()
    model, sgd = build_3vCNN(model)

    model = init_mask_final(model, layerlist, kernelsizelist, 40)

    # 模型训练
    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    tmp_hist = Histories(data=[X_val, Y_val])
    CrossTrain = TrainMethod()
    # 训练
    KernelWeights, MaskWeight = model.layers[0].LossKernel, model.layers[0].MaskFinal
    mu = K.cast(0.0025, dtype='float32')
    lossFunction = ShanoyLoss(KernelWeights, MaskWeight, mu=mu)
    model.compile(loss=lossFunction, optimizer=sgd, metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),patience=20, min_lr=0.0001)
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=modelsave_output_filename.replace(".hdf5", ".checkpointer.hdf5"), verbose=1, save_best_only=True)

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=int(epoch_scheme), shuffle=True,
              validation_data=(X_val, Y_val),
              verbose=2, callbacks=[checkpointer, earlystopper,reduce_lr, tmp_hist, CrossTrain])
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

    tmp_f = open(test_prediction_output, "wb")
    pickle.dump(np.array(report_dic), tmp_f)
    tmp_f.close()
    return [test_auc, modelsave_output_filename]


