# -*- coding: utf-8 -*-
'''
using callback functions to trace the AUCs and losses on each epoch
'''
import keras
from sklearn.metrics import roc_auc_score
# referring to： https://github.com/keunwoochoi/keras_callbacks_example/blob/master/my_callbacks.py
class Histories(keras.callbacks.Callback):
	def __init__(self, data=()):
		super(keras.callbacks.Callback, self).__init__()
		self.x, self.y = data
	def on_train_begin(self,logs={}):
		self.aucs = []
		self.losses = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		y_pred = self.model.predict(self.x)
		tmp_auc = roc_auc_score(self.y, y_pred)
		self.aucs.append(tmp_auc)
		print("tmp_auc: "+str(tmp_auc))
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return
