'''
this file demonstrate how to use layer "Conv1D_IC"
'''

import sys
sys.path.append("../code/")
sys.path.append("../demo/")
from core_conv_ic import Conv1D_IC

import random
import numpy as np
import keras
import pickle
import os
from sklearn.metrics import roc_auc_score


from VCNN_utils import mkdir,check_dir_end,load_dataset
# set limitation to usage of GPU
os.environ["CUDA_VISIBLE_DEVICES"]= "0"


demo_data_path = os.path.join("../data/","simu_data/simu_02/")
demo_result_path = os.path.join("../result/","demo_result/")
mkdir(demo_result_path)


def build_demo_model(number_of_kernel, max_ker_len, init_ker_len,average_IC_update=False,
                     IC_thrd=0.05, input_shape=(1000, 4),verbose = True):
    model_tmp = keras.models.Sequential()
    model_tmp.add(Conv1D_IC(
        input_shape=input_shape,
        kernel_init_len=init_ker_len,
        kernel_max_len=max_ker_len,
        maskTr__IC_threshold=IC_thrd,
        filters=number_of_kernel,
        average_IC_update=average_IC_update,
        verbose = verbose))
    model_tmp.add(keras.layers.GlobalMaxPooling1D())
    model_tmp.add(keras.layers.core.Dropout(0.2))
    model_tmp.add(keras.layers.core.Dense(output_dim=1))
    model_tmp.add(keras.layers.Activation("sigmoid"))
    sgd = keras.optimizers.RMSprop(lr=0.001)  # default 0.001
    model_tmp.compile(loss='binary_crossentropy', optimizer=sgd)
    return model_tmp

# modify the mask length by hand
def basic_demo():
    checkpointer_path = check_dir_end(demo_result_path) + "basic_demo_checkpoint.hdf5"
    # load demo data
    X_train, Y_train = load_dataset(check_dir_end(demo_data_path) + "training_set.hdf5")
    X_test, Y_test = load_dataset(check_dir_end(demo_data_path) + "test_set.hdf5")

    # build demo model
    if len(sys.argv)>1:
        init_ker_len = int(sys.argv[1])
        print("init kernel length: ",init_ker_len)
    model = build_demo_model(number_of_kernel = 128,
                             max_ker_len = 50,
                             init_ker_len = init_ker_len,
                             verbose = True)
    # refer to the Conv1D_IC layer by model.layers[0], "0" means Conv1D_IC is the first layer in the model
    IC_layer = model.layers[0]

    # build callbacks
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=checkpointer_path,
                                                   verbose=1, save_best_only=True)
    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # pre train 10 epoches
    print("first round training")
    model.fit(X_train, Y_train, batch_size=100, nb_epoch=10, shuffle=True, validation_split=0.1,
              verbose=2, callbacks=[checkpointer, earlystopper])
    # load the best checkpoint and start to change mask length
    model.load_weights(checkpointer_path)
    print("second round training")
    for idx in range(5):
        model.fit(X_train, Y_train, batch_size=100, nb_epoch=5, shuffle=True, validation_split=0.1,
                  verbose=2, callbacks=[checkpointer, earlystopper])
        # update mask every epoch
        IC_layer.update_mask()
    # reset the mask's parameters and retrain
    IC_layer.set_mask_stride(1)
    IC_layer.set_IC_threshold(0.5)
    IC_layer.reset_mask_state()
    print("third round training")
    for idx in range(5):
        model.fit(X_train, Y_train, batch_size=100, nb_epoch=5, shuffle=True, validation_split=0.1,
                  verbose=2, callbacks=[checkpointer, earlystopper])
        # update mask every two epoched after
        IC_layer.update_mask()

    # load best model
    model.load_weights(checkpointer_path)
    # test on test set
    print("testing...")
    y_pred = model.predict(X_test)
    test_auc = roc_auc_score(Y_test, y_pred)
    print("auc on test set: ", test_auc)
# a demo to using average IC to update IC threshold at runtime
def update_IC_threshold_demo():
    checkpointer_path = check_dir_end(demo_result_path) + "update_IC_threshold_demo_checkpoint.hdf5"
    # load demo data
    X_train, Y_train = load_dataset(check_dir_end(demo_data_path) + "training_set.hdf5")
    X_test, Y_test = load_dataset(check_dir_end(demo_data_path) + "test_set.hdf5")

    # build demo model
    if len(sys.argv)>1:
        init_ker_len = int(sys.argv[1])
        print("init kernel length: ",init_ker_len)
    model = build_demo_model(number_of_kernel = 128,
                             max_ker_len = 50,
                             init_ker_len = init_ker_len,
                             average_IC_update = True,
                             verbose = True)
    # refer to the Conv1D_IC layer by model.layers[0], "0" means Conv1D_IC is the first layer in the model
    IC_layer = model.layers[0]

    # build callbacks
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=checkpointer_path,
                                                   verbose=1, save_best_only=True)
    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    # pre train 10 epoches
    print("first round training")
    model.fit(X_train, Y_train, batch_size=100, nb_epoch=10, shuffle=True, validation_split=0.1,
              verbose=2, callbacks=[checkpointer, earlystopper])
    # load the best checkpoint and start to change mask length
    model.load_weights(checkpointer_path)
    print("second round training")
    for idx in range(10):
        model.fit(X_train, Y_train, batch_size=100, nb_epoch=5, shuffle=True, validation_split=0.1,
                  verbose=2, callbacks=[checkpointer, earlystopper])
        # update mask every epoch
        IC_layer.update_mask()

    # load best model
    model.load_weights(checkpointer_path)
    # test on test set
    print("testing...")
    y_pred = model.predict(X_test)
    test_auc = roc_auc_score(Y_test, y_pred)
    print("auc on test set: ", test_auc)

if __name__ == '__main__':
    # basic_demo()
    update_IC_threshold_demo()




