# -*- coding: utf-8 -*-
'''
analysis result
'''
import os
import numpy as np
import glob
import h5py
import pdb
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys

sys.path.append("../../corecode/")
from build_models import *
import keras
from keras import backend as K
import pandas as pd
import math
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import gc


# get data_info:
def get_real_data_info(data_root_path):
	return [it.split("/")[-1] + "/" for it in glob.glob(data_root_path + "*")]


def mkdir(path):
	"""
	make dictionary
	:param path:
	:return:
	"""
	isExists = os.path.exists(path)
	if not isExists:
		os.makedirs(path)
		return (True)
	else:
		return False


def load_data(dataset):
	data = h5py.File(dataset, 'r')
	sequence_code = data['sequences'].value
	label = data['labs'].value
	return ([sequence_code, label])


def re_build_model(model, path_root, input_shape):
	Strlist = path_root.split("/")[-1].split("_")
	number_of_kernel = int(Strlist[1].split("-")[1])
	kernel_init_size = int(Strlist[2].split("-")[1])
	max_ker_len = int(Strlist[3].split("-")[1])
	tmp_lst = path_root.split("/")
	if path_root[-1] == "/":
		mode = tmp_lst[-3]
	else:
		mode = tmp_lst[-2]
	if mode == "CNN":
		model = build_CNN_model(model, number_of_kernel, kernel_init_size,
		                        k_pool=1, input_shape=input_shape)
		model.load_weights(path_root)
		print("weight loaded for: " + path_root)

	elif mode == "vCNN" or mode == "vCNNNew" or mode == "vCNN_SEL":
		model = build_vCNN(model, number_of_kernel, max_ker_len, k_pool=1, input_shape=input_shape)[0]
		try:
			model.load_weights(path_root)
		except:
			model.layers[0].trainable_weights =[model.layers[0].kernel,
			                                    model.layers[0].bias, model.layers[0].k_weights]
			model.load_weights(path_root)


		print("weight loaded for: " + path_root)
	else:
		raise ValueError("can't handle mode: " + str(mode))
	return model


###############       restore  kernel    #################

def GeneRateRecoverKerDict(data_root, resultPath, outputPath):
	"""
    generate all kernel
    :param resultPath:
    :return:
    """

	def ModelNameGenerate(ExactModel, ModelTypeStr):
		if ModelTypeStr == "CNN":
			ExactModelNameTem = ExactModel.split("/")[-1].split(".")[0].split("_")
			ExactModelNameT1 = ExactModelNameTem[1] + "_" + ExactModelNameTem[2] \
			                   + "_" + ExactModelNameTem[3]
			ExactModelNameT2 = ExactModelNameT1.replace("-", "_")
			ExactModelReport = glob.glob(
				ExactModel.replace(ExactModel.split("/")[-1],
				                   "Report*" + ExactModelNameT1 + "*"))[0]

			with open(ExactModelReport, "r") as f:
				testAuc = pickle.load(f).tolist()["test_auc"]
			if testAuc < 1:
				ExactModelName = ExactModelNameT2 + "_AUC_" + str(testAuc).replace("0.", "")[:3]
			else:
				ExactModelName = ExactModelNameT2 + "_AUC_1"
		elif ModelTypeStr == "vCNN" or ModelTypeStr == "vCNNNew" or ModelTypeStr == "vCNN_SEL":
			ExactModelNameTem = ExactModel.split("/")[-1].split(".")[0].split("_")
			ExactModelNameT1 = ExactModelNameTem[1] + "_" + ExactModelNameTem[2] + "_" + ExactModelNameTem[3] + "_" + \
			                   ExactModelNameTem[4]
			ExactModelNameT2 = ExactModelNameT1.replace("-", "_")
			ExactModelReport = glob.glob(
				ExactModel.replace(ExactModel.split("/")[-1],
				                   "Report*" + ExactModelNameT1 + "*"))[0]
			with open(ExactModelReport, "r") as f:
				testAuc = pickle.load(f).tolist()["test_auc"]
			if testAuc < 1:
				ExactModelName = ExactModelNameT2 + "_AUC_" + str(testAuc).replace("0.", "")[:3]
			else:
				ExactModelName = ExactModelNameT2 + "_AUC_1"

		else:
			ExactModelName = "vCNN_IC"

		return ExactModelName

	def DB_recover_a_ker(tmp_ker, seqs, labs):
		ker_len = tmp_ker.shape[0]
		count_mat = np.zeros_like(tmp_ker)
		inputs = K.placeholder(seqs.shape)

		ker = K.variable(tmp_ker.reshape(ker_len, 4, 1))
		conv_result = K.conv1d(inputs, ker, padding="valid", strides=1, data_format="channels_last")
		max_idxs = K.argmax(conv_result, axis=1)
		f = K.function(inputs=[inputs], outputs=[max_idxs])
		ret_idxs = f([seqs])[0][:, 0]
		seqlist = []
		for seq_idx, start_idx in enumerate(ret_idxs):
			count_mat = count_mat + seqs[seq_idx, start_idx:start_idx + ker_len, :]
			seqlist.append(seqs[seq_idx, start_idx:start_idx + ker_len, :])
		ret = (count_mat.T / (count_mat.sum(axis=1)).T).T

		del f
		return ret, seqlist

	DataTypelist = glob.glob(resultPath + "/*")
	# KernelsDict = {}
	mode_lst = ["vCNN"]
	i = 0
	for DataType in DataTypelist:
		DataTypeStr = DataType.split("/")[-1]
		training_path = data_root + DataTypeStr + "/train.hdf5"
		X_train, Y_train = load_data(training_path)
		input_shape = X_train[0].shape
		for Modeltype in mode_lst:
			ExactModellist = glob.glob(DataType + "/" + Modeltype + "/*Report*")
			ModelTypeStr = Modeltype.split("/")[-1]

			for ExactModel in ExactModellist:
				ExactModel = glob.glob(ExactModel.replace("Report", "*").replace("pkl", "checkpointer.hdf5"))[0]
				ExactModelName = ModelNameGenerate(ExactModel, ModelTypeStr)
				i = i + 1
				if not os.path.isfile(
						outputPath + DataTypeStr + "/" + ModelTypeStr + "/" + ExactModelName + "/Over.txt"):
					model = keras.models.Sequential()
					kernels = recover_ker(model, ExactModel, ModelTypeStr, input_shape)
					outputPathDBKerTem = outputPath + DataTypeStr + "/" + ModelTypeStr + "/" + ExactModelName + "/" + "DBKer/"
					outputPathKerTem = outputPath + DataTypeStr + "/" + ModelTypeStr + "/" + ExactModelName + "/" + "Ker/"
					outputPathSeqTem = outputPath + DataTypeStr + "/" + ModelTypeStr + "/" + ExactModelName + "/" + "Seqs/"
					mkdir(outputPathKerTem)
					mkdir(outputPathSeqTem)
					mkdir(outputPathDBKerTem)

					if ModelTypeStr == "CNN":
						for ker_id in range(kernels.shape[-1]):
							kernel = kernels[:, :, ker_id]
							if kernel.shape[0] > 0:
								save_data_tem, seqs = DB_recover_a_ker(kernel, X_train, Y_train)
								np.savetxt(outputPathDBKerTem + str(ker_id) + ".txt", save_data_tem)
								np.savetxt(outputPathKerTem + str(ker_id) + ".txt", kernel)

					else:
						for ker_id in range(len(kernels)):
							kernel = kernels[ker_id]
							if kernel.shape[0] > 0:
								save_data_tem, seqs = DB_recover_a_ker(kernel, X_train, Y_train)
								np.savetxt(outputPathDBKerTem + str(ker_id) + ".txt", save_data_tem)
								np.savetxt(outputPathKerTem + str(ker_id) + ".txt", kernel)
					np.savetxt(outputPath + DataTypeStr + "/" + ModelTypeStr + "/" + ExactModelName + "/Over.txt",
					           np.arange(3))
				else:
					print("trained" + str(i))
		gc.collect()


def recover_ker(model, resultPath, modeltype, input_shape):
	"""
    vCNN restore mask
    :param resultPath:
    :param modeltype:
    :param input_shape:
    :return:
    """

	def build_mask(kernel_shape, k_weights):

		def init_left(kernel_shape, k_weights):
			K.set_floatx('float32')
			k_weights_tem_2d_left = np.arange(kernel_shape[0])  # shape[0] length
			k_weights_tem_2d_left = np.expand_dims(k_weights_tem_2d_left, 1)
			k_weights_tem_3d_left = np.repeat(k_weights_tem_2d_left, kernel_shape[2], axis=1) - k_weights[0, :,
			                                                                                    :]  # shape[2] number
			k_weights_3d_left = np.expand_dims(k_weights_tem_3d_left, 1)
			return k_weights_3d_left

		def init_right(kernel_shape, k_weights):
			k_weights_tem_2d_right = np.arange(kernel_shape[0])  # shape[0] length
			k_weights_tem_2d_right = np.expand_dims(k_weights_tem_2d_right, 1)
			k_weights_tem_3d_right = -(
					np.repeat(k_weights_tem_2d_right, kernel_shape[2], axis=1) - k_weights[1, :, :])  # shape[2] number
			k_weights_3d_right = np.expand_dims(k_weights_tem_3d_right, 1)
			return k_weights_3d_right

		def sigmoid(inX):
			return 1.0 / (1 + np.exp(-inX))

		k_weights_3d_left = init_left(kernel_shape, k_weights)
		k_weights_3d_right = init_right(kernel_shape, k_weights)
		k_weights_left = sigmoid(k_weights_3d_left)
		k_weights_right = sigmoid(k_weights_3d_right)
		k_weights = k_weights_left + k_weights_right - 1
		mask = np.repeat(k_weights, 4, axis=1)
		return mask

	def CutKerWithMask(MaskArray, KernelArray):

		CutKernel = []
		for Kid in range(KernelArray.shape[-1]):
			MaskTem = MaskArray[:, :, Kid].reshape(2, )
			leftInit = int(round(max(MaskTem[0], 0), 0))
			rightInit = int(round(min(MaskTem[1], KernelArray.shape[0] - 1), 0))
			kerTem = KernelArray[leftInit:rightInit, :, Kid]
			CutKernel.append(kerTem)

		return CutKernel

	# re load model
	model = re_build_model(model, resultPath, input_shape=input_shape)
	if modeltype == "CNN":
		kernel = K.get_value(model.layers[0].kernel)
	elif modeltype == "vCNN" or modeltype == "vCNNNew" or modeltype == "vCNN_SEL":
		k_weights = K.get_value(model.layers[0].k_weights)
		kernelTem = K.get_value(model.layers[0].kernel)
		# mask = build_mask(kernelTem.shape,k_weights)
		kernel = CutKerWithMask(k_weights, kernelTem)
	else:
		kernel = model.layers[0].get_kernel() * model.layers[0].get_mask()
	del model
	return kernel


#########################################modelAnalysis##################################
#For all trained data sets, kernel extraction, sequence reconstruction
#/home/lijy/VCNNMore/SimulationFinal/AllMoldeKernel original path
#Create Recoverker and MeMeAnalysis respectively. Then give a score
#Create a directory corresponding to each data set in Recoverker, create a model fragment under each data set, and the kernelRebuild directory
# Then create a directory corresponding to each model + auc.
#1.recover reads all kernels, {dataset:{modeltype:{exactmodel:{kernels:[kernels],"seqs":[seqs]}}}}
#2. Restore all kernels using the deepbind method. And save it to the specified directory ~/"dataset"/"modeltype"/"exactmodel"/kernel
#3. Use the deepbind method to find the corresponding seq fragment. And save it to the specified directory ~/"dataset"/"modeltype"/"exactmodel"/sequences
#2 and 3 simultaneously


#########################################caseStudy##################################

def TomTomCompareMotif():
	"""
    Use tomtom for sequence alignment, kernel and motifs are file paths
    :param Kernel:
    :param Motifs:
    :param softwarePath:
    :return:
    """
	softwarePath = "/home/lijy/MotifCompare/meme_4.12.0/src/tomtom"
	Kernelpath = "/home/lijy/VCNNMore/SimulationFinal/AllModelKernelMeMe/"
	DataTypeslist = glob.glob(Kernelpath + "/*")
	mode_lst = ["vCNN", "CNN"]

	for DataTypes in DataTypeslist:
		modellist = glob.glob(DataTypes + "/*")
		for modelTem in mode_lst:
			modelTem = DataTypes + "/" + modelTem
			ExactModellist = glob.glob(modelTem + "/*")
			for ExactModel in ExactModellist:
				Motifspath = "/home/lijy/VCNNMore/SimulationFinal/Data/motifForSimu/MeMeMotif/" \
				             + DataTypes.split("/")[-1] + ".txt"

				Kernel = ExactModel + "/PFM.txt"
				outputPath = ExactModel + "/TomTomS"
				cmdtem = softwarePath + " " + Kernel + " " + Motifspath + " " + "-o" + " " + outputPath
				os.system(cmdtem)


def TomTomMotifStatistic():
	"""
    Use tomtom for sequence alignment, kernel and motifs are file paths
    :param Kernel:
    :param Motifs:
    :param softwarePath:
    :return:
    """
	Kernelpath = "/home/lijy/VCNNMore/SimulationFinal/AllModelKernelMeMe/"
	DataTypeslist = glob.glob(Kernelpath + "/*")
	BestKernel = {}
	mode_lst = ["vCNN", "CNN"]

	for DataTypes in DataTypeslist:
		# modellist = glob.glob(DataTypes + "/*")
		for modelTem in mode_lst:
			modelTem = DataTypes + "/" + modelTem
			ExactModellist = glob.glob(modelTem + "/*")
			for ExactModel in ExactModellist:

				DataSetType = DataTypes.split("/")[-1]
				TomTomResults = pd.read_csv(ExactModel + "/TomTomS/tomtom.txt", sep="\t")
				MotifID = list(set(TomTomResults['Target ID']))
				if DataSetType not in BestKernel.keys():
					BestKernel[DataSetType] = {}
				for motifid in MotifID:
					motifidALL = TomTomResults[TomTomResults['Target ID'] == motifid]
					BKid = motifidALL['E-value'].argmin()
					if motifid not in BestKernel[DataSetType]:
						BestKernel[DataSetType][motifid] = {}
					if modelTem.split("/")[-1] not in BestKernel[DataSetType][motifid]:
						BestKernel[DataSetType][motifid][modelTem.split("/")[-1]] = [
							-math.log(motifidALL['E-value'].min(), 10)]
					else:
						BestKernel[DataSetType][motifid][modelTem.split("/")[-1]].append(
							-math.log(motifidALL['E-value'].min(), 10))

	def draw_bar_plot(data_info, BestKernel, score_type):
		save_p = Kernelpath + data_info + ".png"
		plt.clf()
		mtf_name_lst = list(BestKernel.keys())
		mode_lst = list(BestKernel[mtf_name_lst[0]].keys())

		print("mode_lst", mode_lst)
		mode_num = len(mode_lst)
		for idx in range(mode_num):
			score_lst = []
			mode = mode_lst[idx]
			for mtf_name in mtf_name_lst:
				score_lst.append(BestKernel[mtf_name][mode])
			plt.bar(np.arange(len(score_lst)) + 0.2 * idx, score_lst, width=0.1, label=mode)
			print("score_lst", score_lst)
		plt.xticks(range(len(mtf_name_lst)), size='small')
		plt.title(" ".join([data_info, score_type]))
		plt.legend()
		plt.savefig(save_p)

	def draw_Box_plot(data_info, BestKernel, mtf_name, score_type):
		save_p = Kernelpath + data_info + "_" + mtf_name + ".png"
		plt.clf()
		mode_lst = list(BestKernel.keys())

		print("mode_lst", mode_lst)
		mode_num = len(mode_lst)
		series = {}

		for idx in range(mode_num):
			mode = mode_lst[idx]
			if mode == "vCNN_multi":
				pass
			else:
				if mode == "vCNNNew":
					mode = "vCNN"
					series[mode] = pd.Series(np.asarray(BestKernel["vCNNNew"]))
				else:
					series[mode] = pd.Series(np.asarray(BestKernel[mode]))
		data = pd.DataFrame(series)
		data.boxplot()
		plt.title(" ".join([mtf_name, score_type]))
		plt.legend()
		plt.savefig(save_p)

	for key in BestKernel.keys():
		for key2 in BestKernel[key].keys():
			draw_Box_plot(key, BestKernel[key][key2], key2, score_type="E-value")


def TomTomFileFormatGenerate():
	"""
    load all txt file ，save in the meme format
    :param PwmsPath:
    :return:
    """
	PwmsPath = "/home/lijy/VCNNMore/SimulationFinal/AllMoldeKernel"
	DataTypeslist = glob.glob(PwmsPath + "/*")
	mode_lst = ["vCNN", "CNN"]

	for DataTypes in DataTypeslist:
		mkdir(DataTypes.replace("AllMoldeKernel", "AllModelKernelMeMe"))
		for modelTem in mode_lst:
			modelTem = DataTypes + "/" + modelTem
			mkdir(modelTem.replace("AllMoldeKernel", "AllModelKernelMeMe"))
			ExactModellist = glob.glob(modelTem + "/*")
			for ExactModel in ExactModellist:
				mkdir(ExactModel.replace("AllMoldeKernel", "AllModelKernelMeMe"))
				Kerneltemlist = glob.glob(ExactModel + "/DBKer/*")
				title = open("/home/lijy/VCNNMore/SimulationFinal/result/AnalysisResult/recover_ker/title.txt", 'r')
				f = open(ExactModel.replace("AllMoldeKernel", "AllModelKernelMeMe") + "/PFM.txt", "w")
				for line in title.readlines():
					f.write(line)
				for kernelTem in Kerneltemlist:
					kernel = np.loadtxt(kernelTem)
					f.write("\n")
					kernelname = kernelTem.split("/")[-3] + kernelTem.split("/")[-2] + kernelTem.split("/")[-1]
					kernelname = kernelname.replace(".txt", "")
					f.write("MOTIF" + " " + kernelname + "\n")
					kernelWide = str(kernel.shape[0])
					f.write("letter-probability matrix: alength= 4 w= " + kernelWide + " nsites= 17 E= 4.1e-009\n")
					if len(kernel.shape) > 1:
						for i in range(kernel.shape[0]):
							for j in range(4):
								f.write(str(kernel[i, j]) + "\t")
							f.write("\n")
					else:
						for j in range(4):
							f.write(str(kernel[j]) + "\t")
				f.close()


def TomTomFileFormatMotifGenerate(PwmsPath="/home/lijy/VCNNMore/SimulationFinal/Data/motifForSimu/realMotif"):
	"""
    load all txt file ，save in the meme format
    :param PwmsPath:
    :return:
    """
	ModelTypes = glob.glob(PwmsPath + "/*")
	mkdir(PwmsPath.replace("realMotif", "MeMeMotif"))
	for Dataset in ModelTypes:
		title = open("/home/lijy/VCNNMore/SimulationFinal/result/AnalysisResult/recover_ker/title.txt", 'r')
		f = open(PwmsPath.replace("realMotif", "MeMeMotif") + "/" + Dataset.split("/")[-1] + ".txt", "w")
		for line in title.readlines():
			f.write(line)
		Kerneltemlist = glob.glob(Dataset + "/*.txt")
		for kernelTem in Kerneltemlist:
			kernel = np.loadtxt(kernelTem)
			f.write("\n")
			kernelname = kernelTem.split("/")[-1]
			kernelname = kernelname.replace(".txt", "")
			f.write("MOTIF" + " " + kernelname + "\n")
			kernelWide = str(kernel.shape[0])
			f.write("letter-probability matrix: alength= 4 w= " + kernelWide + " nsites= 17 E= 4.1e-009\n")
			for i in range(kernel.shape[0]):
				for j in range(4):
					f.write(str(kernel[i, j]) + "\t")
				f.write("\n")
		f.close()


if __name__ == '__main__':
	# check the robust
	SimulationDataRoot = "../../Data/ICSimulation/HDF5/"
	SimulationResultRoot = "../../OutPutAnalyse/result/ICSimulation/"
	outputPath = "../../OutPutAnalyse/MotifRebuild/ICSimulation/"

	GeneRateRecoverKerDict(SimulationDataRoot, SimulationResultRoot, outputPath)
	print("GeneRateRecoverKerDict")
	TomTomFileFormatGenerate()
	print("TomTomFileFormatGenerate")
	TomTomCompareMotif()
	print("TomTomCompareMotif")
	TomTomMotifStatistic()
	print("TomTomMotifStatistic")


