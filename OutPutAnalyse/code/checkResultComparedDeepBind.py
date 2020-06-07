# -*- coding: utf-8 -*-

import os
import pickle
import pdb
import numpy as np
import glob
import h5py
import time
import math
import pickle
import pdb
from scipy.stats import wilcoxon, ranksums
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
# from build_models import *
# import keras
# from keras import backend as K
# from vCNN_lg_core import load_kernel_mask
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import scipy.stats as stats

# Get the data_info of the deepbind data set:
# the directory result of the data and result is the same as above
def get_real_data_info(data_root_path):
	return [it.split("/")[-1]+"/" for it in glob.glob(data_root_path+"*")]


# Call this function to flatten the result and return np.array
def flat_record(rec):
	return np.array([x for y in rec for x in y])


# Traverse the path of the simulated dataset, each time calling func

# Get the data_info of the deepbind data set:
# the directory result of the data and result is the same as above

def iter_real_path(func,data_root, result_root):
	'''
	遍历simu数据集，传入func data info 和root_data_dir, root_result_dir
	:param func:
	:param data_root:
	:param result_root:
	:return:
	'''
	data_info_lst = get_real_data_info(data_root)
	ret = {}
	for data_info in data_info_lst:
		ret[data_info] = func(data_root = data_root,result_root=result_root,data_info = data_info)
	return ret


def filelist(path):
	"""
	Generate file names in order
	:param path:
	:return:
	"""

	randomSeedslist = [0, 23, 123, 345, 1234, 9, 2323, 927]

	ker_size_list = [10, 17, 24]
	# ker_size_list = [24]
	number_of_ker_list = [96, 128]
	# number_of_ker_list = [128]

	name = path.split("/")[-2]
	rec_lst = []

	for KernelNum in number_of_ker_list:
		for KernelLen in ker_size_list:
			for random_seed in randomSeedslist:
				if name == "vCNN":
					filename = path + "/Report_KernelNum-" + str(KernelNum) + "_initKernelLen-" + str(
						KernelLen) + "_maxKernelLen-40_seed-" + str(random_seed) \
							   + "_batch_size-100.pkl"
				else:
					filename = path + "/Report_KernelNum-" + str(KernelNum) + "_KernelLen-" + str(
						KernelLen) + "_seed-" + str(random_seed) + "_batch_size-100.pkl"

				rec_lst.append(filename)

	return rec_lst


# reuturn the best auc

def gen_auc_report(data_info,result_root,data_root):
	'''

	:param data_info:
	:param result_root:
	:return:
	'''

	def get_reports(path,datatype):
		aucs = []
		# rec_lst = glob.glob(path+"Report*")
		rec_lst = filelist(path)
		num = 0
		for rec in rec_lst:

			if not os.path.exists(rec):

				continue
			with open(rec,"r") as f:
				tmp_dir = (pickle.load(f)).tolist()
				# aucs.append(flat_record(tmp_dir["auc"]).max())
				global aucouttem, DatatypeAUC
				name = extractUseInfo(rec)

				keylist = aucouttem.keys()
				# if name[-2:] == "24":
				num = num + 1
				aucs.append(tmp_dir["test_auc"])

				if datatype not in DatatypeAUC.keys():
					DatatypeAUC[datatype] = tmp_dir["test_auc"]
				elif DatatypeAUC[datatype] < tmp_dir["test_auc"]:
					DatatypeAUC[datatype] = tmp_dir["test_auc"]

				if name in keylist:
					aucouttem[name].append(tmp_dir["test_auc"])
				else:
					aucouttem[name]=[]
					aucouttem[name].append(tmp_dir["test_auc"])
		if num != 8:
			print(datatype,": ", num)
		# DatatypeAUC[datatype] = np.median(aucs)
		return aucs

	def extractUseInfo(name):
		Knum,KLen = name.split("/")[-1].split("_")[1:3]
		Knum = Knum.replace("-", "")
		KLen = KLen.replace("-", "")
		return Knum+KLen
	model_lst = ["vCNN"]
	ret = {}
	pre_path = result_root + data_info
	for item in model_lst:
		ret[item] = {}
		ret[item]["aucs"] = get_reports(pre_path+item+"/", data_info.replace("/",""))

	for mode in ret:
		tmp_dir = ret[mode]
		aucs = np.array(tmp_dir["aucs"])
		if len(aucs)==0:
			continue
	return ret


# draw the history of AUC，use datainfo as title
# hist_dic is dict, which is like {data_info:{mode:{auc:,loss}}}
# plt_type == “auc” or “loss”
def mkdir(path):
	"""
	Create a directory
	:param path:Directory path
	:return:
	"""
	isExists = os.path.exists(path)
	if not isExists:
		os.makedirs(path)
		return (True)
	else:
		return False

def draw_history(data_info,hist_dic,plt_type):
	save_root = "/home/lijy/VCNNMore/deepbind/AnalysisResult/history/"
	mkdir(save_root)
	color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
	tmp_dic = hist_dic[data_info]
	mode_lst = ["vCNN_lg_multi"]
	print("ploting: "+str(plt_type)+" history:  "+data_info)
	try:
		plt.clf()
		data_info = " ".join(data_info.split("/"))
		plt.title(str(plt_type)+" history:  "+data_info)
		plt.xlabel("epoch")

		plt.ylabel(str(plt_type))

		for idx,mode in enumerate(mode_lst):
			if not (plt_type == "auc" or plt_type == "loss"):
				raise ValueError("cannot support plt_type: "+str(plt_type))
			tmp_data = tmp_dic[mode].tolist()[plt_type]

			# if mode =="vCNN_IC":
			#     pdb.set_trace()
			y = [x for it in tmp_data for x in it]
			# pdb.set_trace()
			plt.plot(np.arange(len(y)),np.array(y),label=mode,color=color_list[idx]) #,label=mode,color=color_list[idx]
		plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
			  fancybox=True, shadow=True, ncol=5)
		plt.savefig(save_root+str(plt_type)+"-"+data_info+".eps", format="eps")
	except:
		pass


def load_data(dataset):
	data = h5py.File(dataset, 'r')
	sequence_code = data['sequences'].value
	label = data['labs'].value
	return ([sequence_code, label])




def GetDeepbindResult(path = "../../OutPutAnalyse/ModelAUC/ChIPSeq/deepbind_pred/*"):
	Filelist = glob.glob(path)
	deepbindDict = {}
	for file_path in Filelist:
		temfile = pd.read_csv(file_path+"/metrics.txt", sep="\t")
		auc = float(temfile.ix[0,0].replace(" ","").replace("auc",""))
		filename = file_path.split("/")[-1].split("_")[0]
		deepbindDict[filename] = auc
	return deepbindDict

def GetCNNResult(name = '1layer_128motif',path = "../../OutPutAnalyse/ModelAUC/ChIPSeq/CNNResult/9_model_result.csv"):
	"""
	Seledt the best models results and output the dict
	:param path:
	:return:
	"""
	file = pd.read_csv(path)
	DictTem = file[['data_set', '1layer_128motif']]
	Dict = DictTem.set_index('data_set').T.to_dict('list')
	DictOutPut = {}

	for keys in Dict.keys():
		DictOutPut[keys] = Dict[keys][0]
	return DictOutPut

######################################################################
def DrawAUC(AUCAixs, AUCdict, statistic_root, ComparedName,data_root="../../Data/ChIPSeqData/HDF5/"):
	"""
	Comparison statistic between 2 models
	:param AUCAixs:
	:return:
	"""
	DatasetALLSize = []
	DatasetWorseCaseSize = []
	WorseDistribute = []
	AUC = [[0,1,1]]
	DataShapeAuc = []
	AUC20000 = [[0,1,1]]
	AUCUnder20000 = [[0,1,1]]
	AUCUp5000 = [[0,1,1]]
	AUCUnder5000 = [[0,1,1]]

	statistic_root = statistic_root + "/"+ ComparedName + "/"
	mkdir(statistic_root)

	for key in AUCAixs:
		aucRes = AUCAixs[key]
		WorseDistribute.append(aucRes)
		data_path = data_root + key
		X_train, Y_train = load_data(data_path + "/train.hdf5")
		DatasetALLSize.append(Y_train.shape[0])
		if AUCAixs[key] < 0:
			DatasetWorseCaseSize.append(Y_train.shape[0])
		DataShapeAuc.append([min(20000,Y_train.shape[0]), AUCAixs[key]])
		AUC.append([Y_train.shape[0], AUCdict[key][0], AUCdict[key][1]])
		if Y_train.shape[0]>20000:
			AUC20000.append([Y_train.shape[0], AUCdict[key][0], AUCdict[key][1]])
		else:
			AUCUnder20000.append([Y_train.shape[0], AUCdict[key][0], AUCdict[key][1]])
			
		if Y_train.shape[0]>5000:
			AUCUp5000.append([Y_train.shape[0], AUCdict[key][0], AUCdict[key][1]])
		else:
			AUCUnder5000.append([Y_train.shape[0], AUCdict[key][0], AUCdict[key][1]])


	########### Dataset Analyse ################################
	WorseDistributeArray = np.asarray(WorseDistribute)
	WorseKey = np.where(WorseDistributeArray < 0)[0]
	WorseKey02 = np.where(WorseDistributeArray < -0.02)[0]

	WorseKeySize = np.asarray(AUC)[WorseKey02,0]
	print(ComparedName + " "+ " Worse0.2 ")
	print(WorseDistributeArray[WorseDistributeArray<-0.02])
	print(WorseKeySize)



	################# ComPared two models##################
	fig, ax = plt.subplots()
	binwidth = abs(min(WorseDistribute))/4
	N, bins, patches = ax.hist(WorseDistribute,bins=np.arange(min(WorseDistribute),
										   max(WorseDistribute) + binwidth, binwidth),
							   edgecolor='white', linewidth=1)
	for i in range(0, 4):
		patches[i].set_facecolor('gray')
	for i in range(4, len(patches)-1):
		if bins[i] < 0.02:
			patches[i].set_facecolor("b")
		elif bins[i] > 0.2:
			patches[i].set_facecolor("r")

	plt.xlabel("Res")  #
	plt.ylabel("Numbers")  #
	plt.title("vCNN AUC minus " + ComparedName +" AUC")
	plt.savefig(statistic_root + "/WorseDistribute.eps", format="eps")
	plt.savefig(statistic_root + "/WorseDistribute.png")
	plt.close()

	ax = plt.subplot()
	ax.boxplot([DatasetALLSize, DatasetWorseCaseSize])
	ax.set_xticklabels(['DatasetALLSize', 'DatasetWorseCaseSize'])
	plt.xlabel("DataSize")  
	plt.savefig(statistic_root + "/DatasetSize.eps", format="eps")
	plt.close()

	ax = plt.subplot()
	ax.boxplot([np.asarray(AUC)[:,1], np.asarray(AUC)[:,2]])
	ax.set_xticklabels([ComparedName, 'VCNN'])
	plt.ylabel("AUC")
	plt.title("AUC Boxplot ")
	plt.savefig(statistic_root + "/boxPlot.eps", format="eps")
	plt.close()

	DataShapeAuc = np.asarray(DataShapeAuc)
	s = [1 for n in range(DataShapeAuc.shape[0])]

	plt.scatter(DataShapeAuc[:,0], DataShapeAuc[:,1],s=s)
	plt.savefig(statistic_root + "/DataShapeAuc.eps", format="eps")
	plt.close()

	################################Draw AUC###########################################
	S = [n[0] for n in AUC]
	AUC = np.asarray(AUC)
	plt.plot([0.5,1],[0.5,1], color='black')
	cm = plt.cm.get_cmap('Paired')
	sc = plt.scatter(AUC[:,1], AUC[:,2], c=S, s=10, cmap=cm)
	plt.colorbar(sc)
	plt.xlabel(ComparedName +" AUC")
	plt.ylabel("vCNN AUC")
	plt.title("AUC Compared with Data Size")
	plt.savefig(statistic_root + "/AUC.eps", format="eps")
	plt.close()

	AUC = np.asarray(AUC)
	plt.plot([0.5,1],[0.5,1], color='black')
	plt.xlabel(ComparedName + " AUC")
	plt.ylabel("vCNN AUC")
	plt.title("AUC Compared")
	plt.scatter(AUC[:, 1], AUC[:, 2], s=10, color="blue")
	plt.savefig(statistic_root + "/AUCNOcolor.png")
	plt.savefig(statistic_root + "/AUCNOcolor.eps", format="eps")
	plt.close()


	S = [n[0] for n in AUC20000]
	AUC = np.asarray(AUC20000)
	plt.plot([0.5,1],[0.5,1], color='black')
	cm = plt.cm.get_cmap('Paired')
	sc = plt.scatter(AUC[:,1], AUC[:,2], c=S, s=10, cmap=cm)
	plt.colorbar(sc)
	plt.xlabel(ComparedName + " AUC")
	plt.ylabel("vCNN AUC")
	plt.title("AUC Compared on Dataset whose Size Larger than 20000")
	plt.savefig(statistic_root + "/AUC20000.eps", format="eps")
	plt.close()
	
	S = [n[0] for n in AUCUnder20000]
	AUC = np.asarray(AUCUnder20000)
	plt.plot([0.5,1],[0.5,1], color='black')
	cm = plt.cm.get_cmap('Paired')
	sc = plt.scatter(AUC[:,1], AUC[:,2], c=S, s=10, cmap=cm)
	plt.colorbar(sc)
	plt.xlabel(ComparedName + " AUC")  # 
	plt.ylabel("vCNN AUC")  # 
	plt.title("AUC Compared on Dataset whose Size Smaller than 20000")
	plt.savefig(statistic_root + "/AUCUnder20000.eps", format="eps")
	plt.close()
	
	S = [n[0] for n in AUCUnder5000]
	AUC = np.asarray(AUCUnder5000)
	plt.plot([0.5,1],[0.5,1], color='black')
	cm = plt.cm.get_cmap('Paired')
	sc = plt.scatter(AUC[:,1], AUC[:,2], c=S, s=10, cmap=cm)
	plt.colorbar(sc)
	plt.xlabel(ComparedName + " AUC")  # 
	plt.ylabel("vCNN AUC")  # 
	plt.title("AUC Compared on Dataset whose Size Smaller than 5000")
	plt.savefig(statistic_root + "/AUCUnder5000.eps", format="eps")
	plt.close()



	S = [n[0] for n in AUCUp5000]
	AUC = np.asarray(AUCUp5000)
	plt.plot([0.5,1],[0.5,1], color='black')
	cm = plt.cm.get_cmap('Paired')
	sc = plt.scatter(AUC[:,1], AUC[:,2], c=S, s=10, cmap=cm)
	plt.colorbar(sc)
	plt.xlabel(ComparedName + " AUC")  # 
	plt.ylabel("vCNN AUC")  # 
	plt.title("AUC Compared on Dataset whose Size Larger than 5000")
	plt.savefig(statistic_root + "/AUCUp5000.eps", format="eps")
	plt.close()


######################################################################
def CompareModels(comparedResult, vCNNAUC, type="DeepBind"):
	"""
	compared the input model and vCNNmodel
	:param comparedResult:
	:param DatatypeAUC:
	:return:
	"""
	AUC = {}
	AUCAixs = {}
	better_number = 0 #
	SignificantBetterNum = 0 #
	bigAxis = [] #
	betterNumber = 0 #
	WorseKeylist = []
	vCNNAUClist = []
	comparedResultlist = []

	for key in comparedResult.keys():
		if key in vCNNAUC.keys():
			AUCAixs[key] = vCNNAUC[key] - comparedResult[key]
			vCNNAUClist.append(vCNNAUC[key])
			comparedResultlist.append(comparedResult[key])
		try:
			AUC[key] = [comparedResult[key], vCNNAUC[key]]
			if vCNNAUC[key] - comparedResult[key] > 0:
				betterNumber = betterNumber + 1
			else:
				WorseKeylist.append(key)
			if comparedResult[key] < 0.6 and vCNNAUC[key] > 0.8:
				better_number = better_number + 1
			if vCNNAUC[key] - comparedResult[key] > 0.02:
				SignificantBetterNum = SignificantBetterNum + 1
			if comparedResult[key] - vCNNAUC[key] > 0.00:
				bigAxis.append(comparedResult[key] - vCNNAUC[key])
		except:
			pass
	print("better 0.2 number:", better_number)
	print("better number:", betterNumber)
	print("SignificantBetterNum:", SignificantBetterNum)
	if len(bigAxis)!=0:
		print("bigAxis:", np.mean(bigAxis))
		print("bigAxismax:", np.max(bigAxis))
		print("bigAxishape:",len(bigAxis))
	stat, p = ranksums(vCNNAUClist, comparedResultlist)
	print("stat:%f", stat)
	print("p-value:" ,p)
	print("MeanVCNN: ", np.mean(vCNNAUClist))
	print(type+": ", np.mean(comparedResultlist))
	return AUC, AUCAixs, WorseKeylist, comparedResultlist

def DrawCompareResult(Keys, vCNNData, CNNData,ComparedNamelist,statistic_root,name):
	"""

	:param Keys:
	:return:
	"""
	CNN = []
	vCNN = []

	DictDraw = {ComparedNamelist[0]:[], ComparedNamelist[1]:[]}

	for key in Keys:
		CNN.append(CNNData[key])
		vCNN.append(vCNNData[key])
		DictDraw[ComparedNamelist[1]].append(CNNData[key])
		DictDraw[ComparedNamelist[0]].append(vCNNData[key])

	mkdir(statistic_root)

	# sns.set_style("darkgrid")

	plt.plot([0.5,1],[0.5,1], color='black')
	plt.xlabel("AUC for "+ name)
	plt.ylabel("AUC for vCNN-based models")

	plt.title("AUC Compared")
	plt.scatter(CNN, vCNN, s=10, color="blue")
	plt.tight_layout()
	plt.savefig(statistic_root + "/" + ComparedNamelist[1]+"Scatter.eps", format="eps")
	plt.savefig(statistic_root + "/" + ComparedNamelist[1] +"Scatter.png")
	plt.close()

	##### barplot
	DictDrawData = pd.DataFrame(DictDraw)
	# sns.set_style("darkgrid")
	pvalue = stats.mannwhitneyu(DictDraw[ComparedNamelist[1]], DictDraw[ComparedNamelist[0]], alternative="less")[1]

	plt.title("P-value: " + format(pvalue, ".2e"), fontsize=20)
	plt.ylabel("AUC", fontsize=20)
	plt.ylim([0.5, 1])
	sns.barplot(data=DictDrawData, capsize=.2, color="w")

	plt.tight_layout()

	plt.savefig(statistic_root + "/" + ComparedNamelist[1]+"Barplot.eps", format="eps")
	plt.savefig(statistic_root + "/" + ComparedNamelist[1] +"Barplot.png")
	plt.close()

	
def WriteWorseKey(filepath, data):
	"""
	
	:param filepath:
	:param data:
	:return:
	"""

	file = open(filepath, 'w')
	for key in data:
		file.write(str(key))
		file.write('\n')
	file.close()
	
def GetDataSize(path):
	"""
	:param path:
	:return:
	"""
	
	datalist = glob.glob(path+"/*")
	dataNumdict = {}
	dataNumlist = []

	for data in datalist:
		f= h5py.File(data+"/train.hdf5")
		Name = data.split("/")[-1]
		num = f["sequences"].shape[0]
		dataNumdict[Name] = num
		dataNumlist.append(num)
		f.close()
		
	return dataNumdict, dataNumlist


def StaWorseKey(WorseKey,dataNumdict,dataNumlist, statisticRoot):
	"""

	:param WorseKey:
	:param dataNumdict:
	:param dataNumlist:
	:param statisticRoot:
	:return:
	"""
	WorseKeySize = []
	for key in WorseKey:
		WorseKeySize.append(dataNumdict[key])
	outputRoot = statisticRoot+"/worseData/"
	mkdir(outputRoot)
	MaxDataSize = np.max(dataNumlist)
	
	sns.set(color_codes=True)
	plt.ylabel("Count of datasets")
	plt.xlabel("Data size")
	plt.xlim(0, MaxDataSize)
	sns.distplot(dataNumlist, bins=40, kde=False)
	plt.savefig(outputRoot +"/" + "DataSize.eps", format="eps")
	plt.savefig(outputRoot + "/" + "DataSize.png")
	plt.close()
	
	sns.set(color_codes=True)
	plt.ylabel("Count of datasets")
	plt.xlabel("Data size")
	plt.xlim(0, MaxDataSize)
	sns.distplot(WorseKeySize, bins=40, kde=False)
	plt.savefig(outputRoot + "/" + "DataSizeWorseCase.eps", format="eps")
	plt.savefig(outputRoot + "/" + "DataSizeWorseCase.png")
	plt.close()

	print("datasize Compare: ",prob_smaller(WorseKeySize, dataNumlist))



def SmallSizeDataAnalyse(dataNumdict, DatatypeAUC, CNNResult, OutputPath):
	"""

	:param dataNumdict:
	:param DatatypeAUC:
	:param CNNResult:
	:param OutputPath:
	:return:
	"""
	VCNNauclist = []
	CNNauclist = []
	dataSizelist = []
	Namelist = []
	for name in dataNumdict:
		if dataNumdict[name]<500:
			VCNNauclist.append(DatatypeAUC[name])
			CNNauclist.append(CNNResult[name])
			dataSizelist.append(dataNumdict[name])
			Namelist.append(name)
	WriteWorseKey("../../TrainCode/ChIpSeqLess500/WorseKeyCNN.txt", Namelist)
	dataSizelist = np.asarray(dataSizelist)/50
	plt.plot([0.5,1],[0.5,1], color='black')
	plt.xlabel("CNN1layers128motifs AUC")
	plt.ylabel("VCNN  AUC")
	plt.title("AUC Compared")
	plt.scatter(CNNauclist, VCNNauclist, s=dataSizelist, color="blue")
	plt.savefig(OutputPath +"/worseData/SmallSizevCNNCNN.eps", format="eps")
	plt.savefig(OutputPath + "/worseData//SmallSizevCNNCNNWorseCase.png")
	plt.close()

def get_Worsedata_info():
	Outputlist = []
	file = open('../../TrainCode/ChIpSeqworseData/WorseKeyCNN.txt', 'r')
	data_list = file.readlines()
	for line in data_list:
		data_info = line.replace("\n", "")
		Outputlist.append(data_info)
	return Outputlist

def prob_smaller(a,b):
	L = len(b)*1.
	lst = []
	for it_a in a:
		lst.append(len([it_b for it_b in b if it_a<it_b])/L)

	return np.array(lst).mean()

if __name__ == '__main__':
	# data path and result path.
	import pandas as pd
	deepbind_data_root = "../../Data/ChIPSeqData/HDF5/"
	result_root = "../../OutPutAnalyse/result/ChIPSeq/"
	# result_root = "/rd2/lijy/KDD/oldResult/result/ChIPSeq//"

	statisticRoot = "../../OutPutAnalyse/ModelAUC/ChIPSeq/"

	mkdir(statisticRoot)
	################################AUC处理###########################################

	aucouttem={}
	DatatypeAUC = {}
	r = iter_real_path(gen_auc_report, data_root=deepbind_data_root, result_root=result_root)

	# compared with deepbind
	deepbindResult = GetDeepbindResult()

	AUC, AUCAixs, WorseKeyDB, DeepBindlist = CompareModels(deepbindResult, DatatypeAUC,"deepbind")

	print("DeepBind finished")
	# Compared with CNN
	CNNResult = GetCNNResult()

	AUC, AUCAixs, WorseKeyCNN, CNNlist = CompareModels(CNNResult, DatatypeAUC,"CNN")

	print("DeepBind finished")

	### all pic

	DrawCompareResult(DatatypeAUC.keys(), DatatypeAUC, CNNResult,
					 ["vCNN","CNN1layers128motifs"],statisticRoot+"/Pic","CNN-based model from Zeng, et al., 2016")

	DrawCompareResult(DatatypeAUC.keys(), DatatypeAUC, deepbindResult,
					 ["vCNN","DeepBind"],statisticRoot+"/Pic","DeepBind models")



	# Worse Analysis

	# WorseKeyCNN = get_Worsedata_info()
	DrawCompareResult(WorseKeyCNN, DatatypeAUC, CNNResult,
					 ["vCNN", "CNN1layers128motifs"],statisticRoot+"/Pic/WorseCNN/", "CNN-based model from Zeng, et al., 2016")

	# DrawCompareResult(WorseKeyDB, deepbindResult, CNNResult,
	# 				 ["DeepBind","CNN1layers128motifs"],statisticRoot+"/Pic/WorseDeepBind/")






	Worselist = list(set(WorseKeyDB).union(set(WorseKeyCNN)))
	
	WriteWorseKey("../../TrainCode/ChIpSeqworseData/WorseKey.txt", Worselist)
	WriteWorseKey("../../TrainCode/ChIpSeqworseData/WorseKeyCNN.txt", WorseKeyCNN)


	stat, p = ranksums(CNNlist, DeepBindlist)
	print("DeepBind CNN", p)
	
	#########staWroseKey######
	
	dataNumdict, dataNumlist = GetDataSize(deepbind_data_root)
	StaWorseKey(WorseKeyCNN, dataNumdict, dataNumlist, statisticRoot+"/Pic/")
	SmallSizeDataAnalyse(dataNumdict, DatatypeAUC, CNNResult, statisticRoot+"/Pic/")
