# vCNNFinal
Implement VCNN layer in class VConv1D_lg.

# Environment requirement
  keras 2.2.4, tensorflow 1.3.0
  
  
# For committee
To repeat graphs and data showed in manuscript, please refer to the [guideline](https://github.com/gao-lab/vCNNFinal/blob/master/corecode/README.md)


# Quick start

Class VConv1D_lg is a keras layer. It can adapt its kernel length at runtime.

The class is implemented at [./corecode/vCNN_lg_core.py](https://github.com/gao-lab/vCNNFinal/blob/master/corecode/vCNN_lg_core.py)

from vCNN_lg_core import VConv1D_lg


When using the layer, you need  pass at least 2 parameters: filers and kernel_init_len.

filers" is the number of kernel. "kernel_init_len" is the initial kernel length.

As following demonstrated, vCNN_lg_core can be added as normal layers to the model. 
(If it is the first layer, parameter "input_shape" is required.)

```model_tmp = keras.models.Sequential()
model_template.add(VConv1D_lg(
        input_shape=input_shape,
        kernel_size=(kernel_init_len),
        filters=number_of_kernel,
        padding='same',
        strides=1))
```		
		
# Run demo code

Clone this repository and run demo code


# under dir ./demo/

```
python Demo.py
```
# you can change the init parameter in Demo.py


