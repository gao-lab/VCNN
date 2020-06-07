# encoding: UTF-8
import os
import pdb
import keras
import h5py
import numpy as np
import pandas as pd
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import sys
from keras.regularizers import l1
from keras.callbacks import LearningRateScheduler
import glob
import tensorflow

from keras.layers.convolutional import *
import pickle
import sklearn.metrics as Metrics
import keras.backend.tensorflow_backend as KTF

#最新版的VCNN，mask只用两个参数决定。
"""
这里我们打算构建一种针对VCNN_lg的特殊训练模式，首先训练kernel，kernel收敛以后训练mask，mask收敛以后。kernel和mask同步训练
于是训练部分分为三部分：
1.训练kernel
2.训练mask
3.合在一起训练
keras自带的train_able dict决定了哪些参数可以训练，但是每次新增加的trainable参数都会直接添加到最后不利于最终的输出，我们将设计函数，在每次训练后的第三部分，把参数调整
成[kernel, bias, k_weights]和初始化的一模一样。方便以后直接构建model和load_model。

"""


class VConv1D_lg(Conv1D):
    """docstring for VConv1D"""

    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super(Conv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k_weights_3d_left = K.cast(0, dtype='float32')
        self.k_weights_3d_right = K.cast(0, dtype='float32')
        self.MaskSize = 0
        self.KernerShape = ()
        self.MaskFinal = 0
        self.KernelSize = 0
        self.LossKernel = K.zeros(shape=self.kernel_size + (4, self.filters))

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        k_weights_shape = (2,) + (1, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.k_weights = self.add_weight(shape=k_weights_shape,
                                         initializer=self.kernel_initializer,
                                         name='k_weights',
                                         regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=keras.initializers.Zeros(),
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def init_left(self):
        K.set_floatx('float32')
        self.k_weights[0, :, :] = K.cast(int(self.KernelSize - self.KernelSize/4), dtype='float32')
        k_weights_tem_2d_left = K.arange(self.kernel.shape[0])  # shape[0]是长度
        k_weights_tem_2d_left = tf.expand_dims(k_weights_tem_2d_left, 1)
        k_weights_tem_3d_left = K.cast(K.repeat_elements(k_weights_tem_2d_left, self.kernel.shape[2], axis=1),
                                       dtype='float32') - self.k_weights[0, :, :]  # shape[2]是numbers
        self.k_weights_3d_left = tf.expand_dims(k_weights_tem_3d_left, 1)

    def init_right(self):
        self.k_weights[0, :, :] = K.cast(int(self.KernelSize + self.KernelSize/4), dtype='float32')
        k_weights_tem_2d_right = K.arange(self.kernel.shape[0])  # shape[0]是长度
        k_weights_tem_2d_right = tf.expand_dims(k_weights_tem_2d_right, 1)
        k_weights_tem_3d_right = -(K.cast(K.repeat_elements(k_weights_tem_2d_right, self.kernel.shape[2], axis=1),
                                          dtype='float32') - self.k_weights[1, :, :])  # shape[2]是numbers
        self.k_weights_3d_right = tf.expand_dims(k_weights_tem_3d_right, 1)

    def regularzeMask(self, maskshape, slip):

        Masklevel = keras.backend.zeros(shape=maskshape)
        for i in range(slip):
            TemMatrix = K.sigmoid(self.MaskSize-float(i)/slip * maskshape[0])
            Matrix = K.repeat_elements(TemMatrix, maskshape[0], axis=0)

            MatrixOut = tf.expand_dims(Matrix, 1)
            Masklevel = Masklevel + MatrixOut
        Masklevel = Masklevel/float(slip) + 1
        return Masklevel


    def call(self, inputs):
        if self.rank == 1:
            # 生成和以前初值shape一样的mask_tem，。
            self.init_left()
            self.init_right()
            k_weights_left = K.sigmoid(self.k_weights_3d_left)
            k_weights_right = K.sigmoid(self.k_weights_3d_right)
            MaskFinal = k_weights_left + k_weights_right - 1
            mask = K.repeat_elements(MaskFinal, 4, axis=1)
            self.MaskFinal = K.sigmoid(self.k_weights_3d_left) + K.sigmoid(self.k_weights_3d_right) - 1
            kernel = self.kernel * mask
            outputs = K.conv1d(
                inputs,
                kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = super(Conv1D, self).get_config()
        config.pop('rank')
        config.pop('data_format')
        return config

class TrainMethod(keras.callbacks.Callback):
    """
    mask and kernel train crossover
    """
    def on_epoch_begin(self, epoch, logs={}):
        oddTrain = [self.model.layers[0].k_weights, self.model.layers[0].bias]
        odd_non_Train = [self.model.layers[0].kernel]

        evenTrain = [self.model.layers[0].kernel, self.model.layers[0].bias]
        even_non_Train = [self.model.layers[0].k_weights]
        AllTrain = [self.model.layers[0].kernel, self.model.layers[0].k_weights, self.model.layers[0].bias]
        All_non_Train = []
        K.set_value(self.model.layers[0].LossKernel, K.get_value(self.model.layers[0].kernel))
        if epoch <= 5:
            self.model.layers[0].trainable_weights = evenTrain
            self.model.layers[0].non_trainable_weights = even_non_Train
        else:
            self.model.layers[0].trainable_weights = AllTrain
            self.model.layers[0].non_trainable_weights = All_non_Train

def ShanoyAveLoss(KernelWeights, MaskWeight, mu, Val=K.cast(0.1, dtype='float32')):
    """
    构建具有香农熵的损失函数
    :param KernelWeights:
    :param MaskWeight:
    :param mu:
    :param Val:
    :return:
    """

    def DingYTransForm(KernelWeights):
        """
        根据丁阳的算法生成PWM
        :param KernelWeights:
        :return:
        """
        ExpArrayT = K.exp(KernelWeights * K.log(K.cast(2, dtype='float32')))
        ExpArray = K.sum(ExpArrayT, axis=1, keepdims=True)
        ExpTensor = K.repeat_elements(ExpArray, 4, axis=1)
        PWM = tf.divide(ExpArrayT, ExpTensor)

        return PWM

    def CalShanoyE(PWM):
        """
        计算PWM的香农熵

        :param PWM:
        :return:
        """
        Shanoylog = -K.log(PWM) / K.log(K.cast(2, dtype='float32'))
        ShanoyE = K.sum(Shanoylog * PWM, axis=1, keepdims=True)
        # 计算均值和两倍标准差
        ShanoyMean = tf.divide(K.sum(ShanoyE, axis=0, keepdims=True), K.cast(ShanoyE.shape[0], dtype='float32'))
        ShanoyMeanRes = K.repeat_elements(ShanoyMean, ShanoyE.shape[0], axis=0)

        return ShanoyE, ShanoyMeanRes

    def lossFunction(y_true,y_pred):

        loss = keras.losses.binary_crossentropy(y_true, y_pred)
        PWM = DingYTransForm(KernelWeights)
        ShanoyE,ShanoyMeanRes = CalShanoyE(PWM)
        # 越往两边贡献越小
        MaskValue = K.cast(0.25, dtype='float32') - (MaskWeight - K.cast(0.5, dtype='float32')) * (MaskWeight - K.cast(0.5, dtype='float32'))
        ShanoylossValue= K.sum((ShanoyE * MaskValue - K.cast(0.25, dtype='float32') * ShanoyMeanRes)
                               * (ShanoyE * MaskValue - K.cast(0.25, dtype='float32') * ShanoyMeanRes)
                               )
        loss += mu * ShanoylossValue
        return loss

    return lossFunction

def ShanoyLoss(KernelWeights, MaskWeight, mu, Val=K.cast(0.1, dtype='float32')):
    """
    构建具有香农熵的损失函数
    :param KernelWeights:
    :param MaskWeight:
    :param mu:
    :param Val:
    :return:
    """

    def DingYTransForm(KernelWeights):
        """
        根据丁阳的算法生成PWM
        :param KernelWeights:
        :return:
        """
        ExpArrayT = K.exp(KernelWeights * K.log(K.cast(2, dtype='float32')))
        ExpArray = K.sum(ExpArrayT, axis=1, keepdims=True)
        ExpTensor = K.repeat_elements(ExpArray, 4, axis=1)
        PWM = tf.divide(ExpArrayT, ExpTensor)

        return PWM

    def CalShanoyE(PWM):
        """
        计算PWM的香农熵

        :param PWM:
        :return:
        """
        Shanoylog = -K.log(PWM) / K.log(K.cast(2, dtype='float32'))
        ShanoyE = K.sum(Shanoylog * PWM, axis=1, keepdims=True)
        # 计算均值和两倍标准差
        ShanoyMean = tf.divide(K.sum(ShanoyE, axis=0, keepdims=True), K.cast(ShanoyE.shape[0], dtype='float32'))
        ShanoyMeanRes = K.repeat_elements(ShanoyMean, ShanoyE.shape[0], axis=0)

        return ShanoyE, ShanoyMeanRes

    def lossFunction(y_true,y_pred):

        loss = keras.losses.binary_crossentropy(y_true, y_pred)
        PWM = DingYTransForm(KernelWeights)
        ShanoyE,ShanoyMeanRes = CalShanoyE(PWM)
        # 越往两边贡献越小
        MaskValue = K.cast(0.25, dtype='float32') - (MaskWeight - K.cast(0.5, dtype='float32')) * (MaskWeight - K.cast(0.5, dtype='float32'))
        ShanoylossValue= K.sum((ShanoyE * MaskValue - K.cast(0.3, dtype='float32'))
                               * (ShanoyE * MaskValue - K.cast(0.3, dtype='float32'))
                               )
        loss += mu * ShanoylossValue
        return loss

    return lossFunction

class KMaxPooling(Layer):
    def __init__(self, K, mode=0, **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.K = K
        self.mode = mode

    def compute_output_shape(self,input_shape):
        shape = list(input_shape)
        shape[1] = self.K
        return tuple(shape)

    def call(self,x):
        k = K.cast(self.K, dtype="int32")
        #sorted_tensor = K.sort(x, axis=1)
        #output = sorted_tensor[:, -k:, :]
        if self.mode == 0:
          output = tensorflow.nn.top_k(tensorflow.transpose(x,[0,2,1]), k)
        elif self.mode ==1:
          output = tensorflow.nn.top_k(x, k)
        else:
          print("not support this mode: ",self.mode)
        return output.values

    def get_config(self):
        config = {"pool_size": self.K}
        base_config = super(KMaxPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def get_mask(model):
    param = model.layers[0].get_weights()
    return param[1]

def get_kernel(model):
    param = model.layers[0].get_weights()
    return param[0]

def init_mask_final(model, init_len_dict, KernelLen):
    """

    :param model:
    :param init_len:初始化的长度对应的数目dict格式，对应长度以及对应长度的数量
    :return:
    """
    param =model.layers[0].get_weights()
    k_weights_shape = param[1].shape
    k_weights = np.zeros(k_weights_shape)
    init_len_list = init_len_dict.keys()
    index_start = 0#记录起始点
    for init_len in init_len_list:
        init_num = init_len_dict[init_len]
        init_len = int(init_len) + 4
        init_part_left = np.zeros([1, k_weights_shape[1], init_num]) + (KernelLen - init_len) / 2
        init_part_right = np.zeros((1, k_weights_shape[1], init_num))+ (KernelLen + init_len)/2
        k_weights[0,:,index_start:(index_start+init_num)] = init_part_left
        k_weights[1,:,index_start:(index_start+init_num)] = init_part_right
        index_start = index_start + init_num
    param[1] = k_weights
    model.set_weights(param)
    return model

def load_kernel_mask(model_path, conv_layer=None):
    param_file = h5py.File(model_path)
    param = param_file['model_weights']['v_conv1d_1']['v_conv1d_1']

    k_weights = param[param.keys()[1]].value

    kernel = param[param.keys()[2]].value

    mask_left_tem = np.repeat(np.arange(kernel.shape[0]).reshape(kernel.shape[0],1), 4, axis=1)
    mask_right_tem = np.repeat(np.arange(kernel.shape[0]).reshape(kernel.shape[0],1), 4, axis=1)
    mask = np.zeros(kernel.shape)
    for i in range(kernel.shape[2]):
        mask_left = np.zeros(mask_left_tem.shape)
        mask_right = np.zeros(mask_right_tem.shape)
        for j in range(mask_left_tem.shape[0]):
            for k in range(mask_left_tem.shape[1]):
                mask_left[j,k] = sigmoid(mask_left_tem[j,k] - mask[0,:,i])
                mask_right[j,k] = sigmoid(-mask_right_tem[j,k] + mask[1,:,i])
        mask[:,:,i] =mask_left + mask_right -1



    return kernel, k_weights, mask

if __name__ == '__main__':

    pass