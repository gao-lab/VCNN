# VCNN
Implement VCNN layer in class Conv1D_IC. 

## Environment requirement

keras 2.0.6, numpy 1.14.2

## For NIPS committee

To repeat graphs and data showed in manuscript, please refer to the [guideline](code/readme.md)


## Quick start

Class Conv1D_IC is a keras layer. It can adapt its kernel length at runtime.

The class is implemented at [./code/core_conv_ic.py](code/core_conv_ic.py)

```
from core_conv_ic import Conv1D_IC
```

In order to initiate the class, it requires to pass at least 2 parameters: filers and kernel_init_len. 

"filers" is the number of kernel. "kernel_init_len" is the initial kernel length.

As following demonstrated, Conv1D_IC can be added as normal layers to the model. (If it is the first layer, parameter "input_shape" is required.) 

```
model_tmp = keras.models.Sequential()
model_tmp.add(Conv1D_IC(
    input_shape=input_shape,
    kernel_init_len=init_ker_len,
    filters=number_of_kernel)) 
```



## Run demo code

Clone this repository and run demo code

```
# under dir ./demo/
ipython demo.py 16
# 16 is the init length of kernel
```

## Unit test

Unit test is in [./test/uni_test.py](test/uni_test.py).

```
The test will cover the following:
   （a）Test_mask_update:
   		Test the method "update_mask()".
   （b）Test_training
   		General test the training and saving of the model
   （c）Test_saving
   		Test "load_weights()" function. (Have to run Test_training first.)
```
