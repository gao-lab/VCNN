# Usage of Conv1D_IC layer

```
class Conv1D_IC(Conv1D)
```

Conv1D_IC is 1D convolution layer with adaptable kernel length. The implementation is based on:

 keras 2.0.6,

 numpy 1.14.2

The code support both Tensorflow and Theano backends

```
import keras
from core_conv_ic import Conv1D_IC

model = keras.models.Sequential()

# define the hyper parameters
number_of_kernel = 128 
max_ker_len = 50	
init_ker_len = 16
IC_thrd=0.05 
input_shape=(1000, 4)
verbose = True

# use the Conv1D_IC as a layer to build Neural network
model.add(Conv1D_IC(
    input_shape=input_shape,
    kernel_init_len=init_ker_len,
    kernel_max_len=max_ker_len,
    maskTr__IC_threshold=IC_thrd,
    filters=number_of_kernel,
    verbose = verbose))

# add more layers according to your need
model.add(keras.layers.GlobalMaxPooling1D())
model.add(keras.layers.core.Dropout(0.2))
model.add(keras.layers.core.Dense(output_dim=1))
model.add(keras.layers.Activation("sigmoid"))
sgd = keras.optimizers.RMSprop(lr=0.001)  # default 0.001
model.compile(loss='binary_crossentropy', optimizer=sgd)
```

The code above is a basic demo to use Conv1D_IC as a layer to build a model. A few hyper parameters are needed to initiate the layer.

|         Name         |                 Meaning                  |
| :------------------: | :--------------------------------------: |
|     input_shape      | An tuple. The shape of input data. This value can be calculated automatically if this is not the first layer of the model |
|   kernel_init_len    | An integer. The initial length of kernel |
|    kernel_max_len    |   An integer. The max length of kernel   |
| maskTr__IC_threshold | A float. The threshold for updating the mask edges |
|       filters        |    An integer. The number of kernels     |
|       verbose        | A bool. If print the mask updating information in the console |

```
# refer to the Conv1D_IC layer by model.layers[0], 
# "0" means Conv1D_IC is the first layer in the model
IC_layer = model.layers[0]
```

The code above use "IC_layer" to refer to the Conv1D_IC layer in the model, the at runtime, can use the methods in Conv1D_IC class. Following methods are supported:

|             Methods             |                Functions                 |
| :-----------------------------: | :--------------------------------------: |
|          update_mask()          | When calling this function, the kernel lengths in Conv1D_IC will adjust for a time. |
|   set_mask_stride(new_stride)   | Set the mask moving stride to a new value. |
| set_IC_threshold(new_threshold) | Set a new IC threshold for updating mask |
|       reset_mask_state()        |  Reset all the edges' state to "start"   |

Below is a demo of how to use these method for training and testing

```
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
```

In the demo above, the model will first be trained for 10 epochs. The will update the mask every 5 epochs and update 5 times. During this first round updating mask, the stride is 3 and IC_threshold is 0.05. Then for the second round, change the stride to 1 and IC_threshold to 0.5 and load the best checkpoint. Also update every 5 epochs and update for 5 times.

This is a demo implementation. There are many other ways to do so, e.g. writing the update in a callback function.

The overall demo can be found in "demo.py". 

## An updated version:

Using kerenl's average IC as IC threshold. Run the demo code, modify "demo.py" as following:

```
if __name__ == '__main__':
    # basic_demo()
    update_IC_threshold_demo()
```

In this new version, class Conv1D_IC(Conv1D) has a new parameter to be set: "average_IC_update". It's a bool variable, the VCNN layer will use average IC in valid part of kernel to update IC_threshold.

In this case, the training process can be more simplified. It should include 2 rounds training. First pretrain and then update the mask.

```
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
```