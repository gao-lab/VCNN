# -*- coding: utf-8 -*-
"""Core Keras ConvIC classes
"""
import keras
from keras import backend as K
from keras.layers import Conv1D
from keras.utils import conv_utils
from keras.engine import InputSpec
import numpy as np
import warnings
import math
from keras.engine import Layer
import tensorflow
from pprint import pprint

class Conv1D_IC(Conv1D):
    '''1D Convolution layer supports changing the valid length of kernel in run time
    This keras convolution layer supports changing the valid length of kernel by
    using a one-zero mask to multiply the kernel. The length of ones in mask represent
    the valid part for each kernel.
    During the training time, method "update_mask" can change the valid lengths of each
    mask, by calculating the Information Content of each kernel. There are some parameters
    determine the mask as below:
        Each mask is a one-zero matrix, the same size of the max kernel length. As for
        the kernel in 1D sequence Detection, the kernel has the shape of (kernel_size,4,filters).
        For the i-th kernel (kernel[:,:,i]) and the corresponding mask is mask[:,:,i].
        Each mask has a "head" and a "tail", which are zero sub-matrix, at the first and the last
        few rows. And the the columns in the middle is ones matrix. There are two parameters
        describe them: "length" and "state". "length" is the number of rows for head or tail and
        "state" is to describe how the head or tail moving in the runtime.
            mask__head_state_lst: a list of string. Each indicates a length of one mask head
            mask__tail_state_lst: a list of string. Each indicates a length of one mask tail
            mask__head_length_lst: a list of int. Each indicates a length of one mask head
            mask__tail_length_lst: a list of int. Each indicates a length of one mask tail
        When doing the convolution, it will multiply the mask to the kernel, leaving the "invalid"
        part of the kernel zero.
        During the train time, using method "update_mask" to change the valid kernel length. The
        algorithm for updating the mask can be found in reference

        # Argument:
            filters: the number of kernel
            kernel_init_len: the init length kernel's valid part. By default, the valid part of
                the kernel is placed in the middle of the kernel.
            kernel_max_len: the max length of the kernel (including the valid and invalid part).
                By default is 50
            verbose: a bool, if the message will be printed in the concole. The messages including:
                the masks' states and lengths.
            kernel_min_len: a integer, the minimal size of the kernel. Once the valid kernel
            length reaches this value, will stop this kernel's mask. By default is 4.
            bio__non_IC_vec: the none information content row, in DNA motif detection, means
                the background distribution of bases. Set [0,25,0,25,0,25,0,25] by default
            maskTr__IC_threshold: the threshold of IC during training. By default is 0.05. it is
                chosen according to the IC distribution of the initiated kernel.
            maskTr__norm_base: the base for the normalization. It may change automatically
                according to the kernel length. This modification is make the distribution of init
                IC consist with the pre_calculated one (see supplementary material)
            maskTr__stride_size: an integer, the size of the stride when changing the edge of the mask.
                By default is 3
            maskTr__IC_window_size: an integer, the window size when calculate the IC of the kernel.
                This is to prevent split the motif with a "gap" in the middle. By default is 5
            Other parameters for convolution is set by default. There are different reasons for doing so
            "padding": is set to "same", because VCNN have invalid part of kernel, where the value is zero.
                In order to prevent the edge of each sequence be ignored.
            "dataformat": is set to "channels_last" for the convenience of implementation
                (this can be changed in future version)
            "kernel_initializer": is set to "RandomUniform", in order to calculated the IC threshold's
                initial distribution. Also unnecessary limitation just for the convenience of implementation
            other parameters are chosen only for the implementation convenience. Can be changed in future version
            "average_IC_update": a bool variable, using average IC as threshold when updating mask edges
        # Reference:
            The algorithm is described in doc: {to fix!}
    '''
    def __init__(self,filters,
                 kernel_init_len,
                 kernel_max_len = 50,
                 verbose = True,
                 kernel_min_len=4,
                 bio__non_IC_vec=np.ones(4)*0.25,
                 maskTr__IC_threshold=0.05,
                 maskTr__norm_base=10.,
                 maskTr__stride_size=3,
                 maskTr__IC_window_size=5,
                 average_IC_update = False,
                 **kwargs
                 ):
        self.verbose = verbose
        self.average_IC_update = average_IC_update
        super(Conv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_max_len,
            strides=1,
            padding="same",
            data_format='channels_last',
            dilation_rate = 1,
            activation=None,
            use_bias=True,
            kernel_initializer="RandomUniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs)
        self.input_spec = InputSpec(ndim=3)
        # make sure there is not 0 in bio__non_IC_vec, to prevent divide by zero
        assert not any(bio__non_IC_vec == 0.)
        # set the new parameters

        # 无信号的向量，等价为序列的背景分布
        self.bio__non_IC_vec = bio__non_IC_vec
        # the entropy of non_IC_vec
        self.bio__non_IC_entropy = np.array([-it*math.log(it,2.)
                                             for it in bio__non_IC_vec if not it==0.]).sum()

        # the threshold when changing the mask edges when updating
        self.maskTr__IC_threshold = maskTr__IC_threshold
        # the window size when calculate the IC
        self.maskTr__IC_window_size = maskTr__IC_window_size
        # the stride for moving mask edges
        self.maskTr__stride_size = maskTr__stride_size
        # the base used for normalize mask
        self.maskTr__norm_base = maskTr__norm_base

        # the max length of kernel length
        self.mask__kernel_max_len = kernel_max_len
        # the minimum length of kernel length
        self.mask__kernel_min_len = kernel_min_len
        # the init size (length) of the valid part kernel
        self.mask__kernel_init_len = kernel_init_len
        # a list describe the information of mask's head
        # the i-th item in the list refers to the i-th mask's head state info
        self.mask__head_state_lst = ["start" for idx in range(filters)]
        # a list describe the information of mask's tail
        # the i-th item in the list refers to the i-th mask's tail state info
        self.mask__tail_state_lst = ["start" for idx in range(filters)]

        # calculate the length of the head and the tail, by default is placed the valid kernel
        # in the middle
        self.mask__head_init_len = int(0.5*(self.mask__kernel_max_len
                                            -self.mask__kernel_init_len))
        self.mask__tail_init_len = int(self.mask__kernel_max_len
                                        -self.mask__head_init_len
                                        -self.mask__kernel_init_len)
        # init mask's head and tail length
        self.mask__head_len_lst = None
        self.mask__tail_len_lst = None

        # the temporary kernel value
        self.tmp_kernel_val = None
        # the temporary mask value
        self.tmp_mask_val = None
        # the valid kernel at the beginning
        self.real_kernel_size = conv_utils.normalize_tuple(self.mask__kernel_init_len,
                                                           self.rank,
                                                           'real_kernel_shape')
        return
    # multiply kernel with mask before doing convolution
    def call(self,inputs):
        if self.rank == 1:
            kernel = self.kernel*self.mask # multiply mask and kernel to get the real for convolution
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
    # to support save and load model
    def get_config(self):
        base_config = super(Conv1D, self).get_config()
        config = {
            'bio__non_IC_vec' : self.bio__non_IC_vec,
            'maskTr__IC_threshold' : self.maskTr__IC_threshold,
            'maskTr__IC_window_size' : self.maskTr__IC_window_size,
            'maskTr__stride_size' : self.maskTr__stride_size,
            'maskTr__norm_base' : self.maskTr__norm_base,
            'mask__kernel_max_len' : self.mask__kernel_max_len,
            'mask__kernel_min_len' : self.mask__kernel_min_len,
            'mask__kernel_init_len' : self.mask__kernel_init_len,
            'mask__head_state_lst' : self.mask__head_state_lst,
            'mask__tail_state_lst': self.mask__tail_state_lst,
            'mask__head_init_len' : self.mask__head_init_len,
            'mask__tail_init_len' : self.mask__tail_init_len,
            'mask__head_len_lst' : self.mask__head_len_lst,
            'mask__tail_len_lst' : self.mask__tail_len_lst,
            'tmp_kernel_val' : self.tmp_kernel_val,
            'tmp_mask_val' : self.tmp_mask_val
        }
        return dict(list(base_config.items()) + list(config.items()))
    # init kernel and mask as keras Tensor
    def build(self,input_shape):
        # get the kernel shape to init (including both valid and in valid part)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # mask is not trainable
        self.mask = self.add_weight(shape=kernel_shape,
                                    initializer=self.kernel_initializer,
                                    name='mask',
                                    trainable = False)
        # reset the value for mask
        self.tmp_mask_val = np.ones_like(K.get_value(self.mask))
        self.tmp_mask_val[:self.mask__head_init_len, :, :] = 0
        self.tmp_mask_val[-self.mask__tail_init_len:, :, :] = 0
        K.set_value(self.mask,self.tmp_mask_val)

        # rescale kernel via uni_vias
        self.tmp_kernel_val = K.get_value(self.kernel)

        # following built like Conv1D
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

        return
    # get the value of mask
    def get_mask(self):
        ret = K.get_value(self.mask)
        return ret
    # get the value of kernel
    def get_kernel(self):
        ret = K.get_value(self.kernel)
        return ret
    # set a new stride for moving mask
    def set_mask_stride(self,new_stride_size):
        self.maskTr__stride_size = new_stride_size
        if self.verbose:
            print("new mask moving stride is set: ", self.maskTr__stride_size)
    # set a new IC threshold
    def set_IC_threshold(self,new_IC_thred):
        self.maskTr__IC_threshold = new_IC_thred
        if self.verbose:
            print("new IC threshold is set: ", self.maskTr__IC_threshold)
    # set all the state to "start"
    def reset_mask_state(self):
        for idx in range(self.filters):
            self.mask__head_state_lst[idx] = "start"
            self.mask__tail_state_lst[idx] = "start"
        if self.verbose:
            print("mask rested")
        self.enclosed__show_mask_info()
        return
    # get the edge information of the mask.
    # update mask__head_len_lst and mask__tail_len_lst if they haven't been inited
    # return [head_len_lst,tail_len_lst]
    def enclosed__get_mask_edge(self):
        # update the mask_val and head_len_lst and tail_len_lst
        if not self.mask__head_len_lst:
            # only in here when first init the class
            # also, after loading weight, a new model will also enter here
            self.tmp_mask_val = self.get_mask()
            self.tmp_kernel_val = self.get_kernel()
            self.mask__head_len_lst = []
            self.mask__tail_len_lst = []
            for ker_id in range(self.filters):
                tmp_mask_slide = self.tmp_mask_val[:,1,ker_id]
                tmp_valid_idx = [idx for idx,item in enumerate(tmp_mask_slide) if item == 1]
                tmp_head_len = tmp_valid_idx[0]
                tmp_tail_len = self.mask__kernel_max_len - tmp_valid_idx[-1] - 1
                self.mask__head_len_lst.append(tmp_head_len)
                self.mask__tail_len_lst.append(tmp_tail_len)
            return [self.mask__head_len_lst,self.mask__tail_len_lst]
        else:
            return [self.mask__head_len_lst,self.mask__tail_len_lst]
    # get the state of mask: return [head_state,tail_state]
    def enclosed__get_mask_state(self):
        return [self.mask__head_state_lst,self.mask__tail_state_lst]
    # show the mask info on console: how many a heads and tails have stopped
    def enclosed__show_mask_info(self):
        h_state,t_state = self.enclosed__get_mask_state()
        head_stopped = [idx for idx,item in enumerate(h_state) if item == "stop"]
        tail_stopped = [idx for idx,item in enumerate(t_state) if item == "stop"]
        if self.verbose:
            print("heads: {0} out of {1} have stopped".format(len(head_stopped),len(h_state)))
            print("tails: {0} out of {1} have stopped".format(len(tail_stopped),len(t_state)))
        h,t = self.enclosed__get_mask_edge()
        valid_len = [self.mask__kernel_max_len-h[idx]-t[idx] for idx in range(self.filters)]
        if self.verbose:
            rep_dic = {}
            for it in valid_len:
                if it not in rep_dic:
                    rep_dic[it] = 0
                rep_dic[it] = rep_dic[it]+1
            pprint(rep_dic)
        return len(head_stopped)==len(h_state) and len(tail_stopped)==len(t_state)
    # calculate the IC for the kernel
    # by default, the IC returned is the max value within the window,
    # more features can be implemented in the future:
    # return the average IC in the window for example.
    def helper__call_IC(self,mask_idx,start_idx,end_idx):
        # calculate the IC of a kernel, mask_idx refers to a kernel:
        # start_idx,end_idx locate the interval of the kernel, where to calculated the IC
        tmp_kernel = self.tmp_kernel_val[:,:,mask_idx] # find the kernel
        base = self.maskTr__norm_base
        IC_lst = []
        for idx in range(end_idx-start_idx):
            tmp_arr = np.power(base, tmp_kernel[start_idx + idx, :])
            tmp_p = tmp_arr / tmp_arr.sum()
            IC_lst.append(self.bio__non_IC_entropy -
                          np.array([-it*math.log(it,2.)
                                    for it in tmp_p if not it == 0.]).sum())
        return np.array(IC_lst).max(),np.array(IC_lst).sum()
    # update a mask according to the mask_idx
    # only set the value of tmp_mask here
    # the mask's tensor will be reset after changing have done for every mask
    def helper__update_a_mask(self,mask_idx):
        h_state,t_state = self.enclosed__get_mask_state()
        tmp_head_len_lst,tmp_tail_len_lst = self.enclosed__get_mask_edge()
        # check: if mark the state of head or tail to "stop"
        if h_state[mask_idx] == "stop" and t_state[mask_idx] == "stop":
            # return if both end of the mask is "stop"
            return
        tmp_head_len = tmp_head_len_lst[mask_idx]
        tmp_tail_len = tmp_tail_len_lst[mask_idx]
        tmp_head_state = h_state[mask_idx]
        tmp_tail_state = t_state[mask_idx]
        # stop the head and tail, if kernel has reached it's minimum value
        # and the states of both ends are "backward"
        if (not tmp_head_len + tmp_tail_len + self.mask__kernel_min_len<self.mask__kernel_max_len) \
                and (tmp_head_state == "backward" or tmp_head_state == "stop") \
                and (tmp_tail_state == "backward" or tmp_tail_state == "stop"):
            self.mask__head_state_lst[mask_idx] = "stop"
            self.mask__tail_state_lst[mask_idx] = "stop"
        # change the state of head to "stop" if it's state is "forward" and head length = 0
        if tmp_head_state == "forward" and tmp_head_len == 0:
            self.mask__head_state_lst[mask_idx] = "stop"
        # change the state of tail to "stop" if it's state is "forward" and tail length = 0
        if tmp_tail_state == "forward" and tmp_tail_len == 0:
            self.mask__tail_state_lst[mask_idx] = "stop"

        # before update the mask, update the tmp_mask
        tmp_a_new_mask = np.ones([self.mask__kernel_max_len,4])
        tmp_a_new_mask[:tmp_head_len,:] = 0
        tmp_a_new_mask[-tmp_tail_len:,:] = 0
        self.tmp_mask_val[:,:,mask_idx] = tmp_a_new_mask
        return
    # update the head_state and head length when call this function
    def helper__update_head_state(self,mask_idx):
        # 更新指定mask_idx的head state
        tmp_state = self.mask__head_state_lst[mask_idx]
        tmp_head_len = self.mask__head_len_lst[mask_idx]
        tmp_tail_len = self.mask__tail_len_lst[mask_idx]
        # 对于检查IC来说，head的最大长度，不能大于 max_kernel_len-tmp_tail_len
        # 否则会超出valid kernel的区域
        max_info_len = self.mask__kernel_max_len-tmp_tail_len

        # 对于移动下一步的mask而言，最大的head length 不能覆盖 min_kernel_valid
        max_strite_len = self.mask__kernel_max_len-tmp_tail_len-self.mask__kernel_min_len

        if tmp_state == "stop":
            return
        next_info_head_len = int(min(max_info_len,tmp_head_len+self.maskTr__IC_window_size))
        pre_IC,tot_IC = self.helper__call_IC(mask_idx,tmp_head_len,next_info_head_len)
        if tmp_state == "start":
            if pre_IC > self.maskTr__IC_threshold:
                self.mask__head_state_lst[mask_idx] = "forward"
                self.mask__head_len_lst[mask_idx] =  int(max(0, tmp_head_len - self.maskTr__stride_size))
            else:
                self.mask__head_state_lst[mask_idx] = "backward"
                self.mask__head_len_lst[mask_idx] = \
                    int(min(max_strite_len, tmp_head_len + self.maskTr__stride_size))
        elif tmp_state == "forward":
            if pre_IC > self.maskTr__IC_threshold:
                self.mask__head_len_lst[mask_idx] = int(max(0,tmp_head_len-self.maskTr__stride_size))
            else:
                self.mask__head_state_lst[mask_idx] = "stop"
        elif tmp_state == "backward":
            if pre_IC > self.maskTr__IC_threshold:
                self.mask__head_state_lst[mask_idx] = "stop"
            else:
                self.mask__head_len_lst[mask_idx] =\
                    int(min(max_strite_len, tmp_head_len + self.maskTr__stride_size))
        return
    # update the tail_state and tail length when call this function
    def helper__update_tail_state(self,mask_idx):
        # 更新指定mask_idx的tail state
        tmp_state = self.mask__tail_state_lst[mask_idx]
        tmp_head_len = self.mask__head_len_lst[mask_idx]
        tmp_tail_len = self.mask__tail_len_lst[mask_idx]

        # 对于检查IC来说，tail的最大长度，不能大于 max_kernel_len-tmp_head_len
        # 否则会超出valid kernel的区域
        max_info_len = self.mask__kernel_max_len - tmp_head_len

        # 对于移动下一步的mask而言，最大的tail length 不能覆盖 min_kernel_valid
        max_strite_len = self.mask__kernel_max_len - tmp_head_len - self.mask__kernel_min_len

        if tmp_state == "stop":
            return
        next_info_tail_len = int(min(max_info_len, tmp_tail_len + self.maskTr__IC_window_size))

        pre_IC,tot_IC = self.helper__call_IC(mask_idx,
                             self.mask__kernel_max_len-next_info_tail_len,
                             self.mask__kernel_max_len-tmp_tail_len)
        if tmp_state == "start":
            if pre_IC > self.maskTr__IC_threshold:
                self.mask__tail_state_lst[mask_idx] = "forward"
                self.mask__tail_len_lst[mask_idx] = int(max(0, tmp_tail_len - self.maskTr__stride_size))
            else:
                self.mask__tail_state_lst[mask_idx] = "backward"
                self.mask__tail_len_lst[mask_idx] = int(min(max_strite_len, tmp_tail_len + self.maskTr__stride_size))
        elif tmp_state == "forward":
            if pre_IC > self.maskTr__IC_threshold:
                self.mask__tail_len_lst[mask_idx] = int(max(0, tmp_tail_len - self.maskTr__stride_size))
            else:
                self.mask__tail_state_lst[mask_idx] = "stop"
        elif tmp_state == "backward":
            if pre_IC > self.maskTr__IC_threshold:
                self.mask__tail_state_lst[mask_idx] = "stop"
            else:
                self.mask__tail_len_lst[mask_idx] = int(min(max_strite_len, tmp_tail_len + self.maskTr__stride_size))
        return
    # update_IC threshold, by default using average to update
    def update_IC(self,type = "ave"):
        supp_lst = ["ave"]
        if type not in supp_lst:
            raise NotImplemented("not support using {0} to update IC threshold".format(type))
        tot_IC = 0.
        tot_column = 0.
        # get head_lst and tail_lst of mask
        head_lst,tail_lst = self.enclosed__get_mask_edge()
        for mask_idx in range(self.filters):
            h = head_lst[mask_idx]
            t = self.mask__kernel_max_len-tail_lst[mask_idx]
            tot_column = tot_column + t-h
            tot_IC = tot_IC + self.helper__call_IC(mask_idx,h,t)[1]
        self.maskTr__IC_threshold = tot_IC/tot_column
        if self.verbose:
            print("Using {0} IC to update IC threshold: {1}".format(type,self.maskTr__IC_threshold))
        return
    # update the mask, each time calling this function
    def update_mask(self):
        # get the value of tmp_mask_val and tmp_kernel_val updated first!
        self.tmp_mask_val = K.get_value(self.mask)
        self.tmp_kernel_val = K.get_value(self.kernel)
        # update the edge information {head and tail} + {state and length}
        self.enclosed__get_mask_edge()
        if self.average_IC_update:
            self.update_IC("ave")

        for mask_idx in range(self.filters):
            self.helper__update_head_state(mask_idx)
            self.helper__update_tail_state(mask_idx)
            self.helper__update_a_mask(mask_idx)
        # reset_the value of mask
        K.set_value(self.mask,self.tmp_mask_val)
        # return True when every head and tail are "stop"
        if_stop = self.enclosed__show_mask_info()
        return if_stop
    # use for test, set the kernel to a certain value
    def test__set_kernel(self,val):
        K.set_value(self.kernel,val)
        return
    # use for test, set the kernel to a certain value
    def test__set_mask(self,val):
        K.set_value(self.mask,val)
        return


