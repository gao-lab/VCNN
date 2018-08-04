# encode update mask in keras callback
import keras
from sklearn.metrics import roc_auc_score
class mask_update(keras.callbacks.Callback):
	def __init__(self,conv_ic_layer,start_epoc = 5):
		super(keras.callbacks.Callback, self).__init__()
		self.conv_ic_layer = conv_ic_layer
		self.epoc_num = 0
		self.start_epoc = start_epoc

	def on_train_begin(self,logs={}):
		return

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		print("maskTr__IC_threshold:  ",self.conv_ic_layer.maskTr__IC_threshold)
		print("maskTr__stride_size:  ",self.conv_ic_layer.maskTr__stride_size)
		self.epoc_num = self.epoc_num+1
		if self.epoc_num>self.start_epoc:
			if_stop = self.conv_ic_layer.update_mask()
			if if_stop:
				print("restart mask update")
				self.conv_ic_layer.maskTr__IC_threshold = 0.5
				self.conv_ic_layer.reset_mask_state()
				self.conv_ic_layer.maskTr__stride_size = max(1,self.conv_ic_layer.maskTr__stride_size-1)
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return
