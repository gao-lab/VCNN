import glob
import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.interpolate import spline
plt.switch_backend('agg')

def NameSelect(name):
	"""

	:param name:
	:return:
	"""
	sequencelen = int(name.split("/")[-1].split("_")[0])
	kernellen = int(name.split("/")[-1].split("_")[2])
	pideal = float(name.split("/")[-1].split("_")[1])

	return sequencelen, kernellen, pideal


def Draw(dict, path):
	"""

	:return:
	"""
	keylist = list(dict.keys())
	keylist.sort()
	for key in keylist:
		# if key in ["6", "7", "8"]:
			arr = np.asarray(dict[key]).T
			index = np.argsort(arr[0])
			arr[0] = arr[0][index]
			arr[1] = arr[1][index]
			# xnew = np.linspace(arr[0].min(), arr[0].max(), 300)  # 300 represents number of points to make between T.min and T.max
			# power_smooth = spline(arr[0], arr[1], xnew)
			# plt.plot(xnew,power_smooth, label= key)
			plt.plot(arr[0],arr[1], label= key)
			plt.xlabel("pideal")
			plt.ylabel("preal")
			plt.legend()
	plt.savefig(path+"/test.png")




def Main(path,ker_size_list):
	"""

	:param path:
	:return:
	"""
	resultlist = glob.glob(path + "/*.txt")
	OutputPath = "../VCNNPaperFigure/"

	Kernellendict = {}

	for result in resultlist:
		sequencelen, kernellen, pideal = NameSelect(result)

		if kernellen not in ker_size_list:
			continue
		preal = np.loadtxt(result)[1]

		if int(kernellen) in Kernellendict:
			Kernellendict[int(kernellen)].append([pideal,preal])
		else:
			Kernellendict[int(kernellen)] = [[pideal,preal]]
	Draw(Kernellendict,OutputPath)

if __name__ == '__main__':
	path = "../resultLower/simuMtf_Len-8_totIC-10/"
	ker_size_list = [4, 5, 6, 7, 8, 10, 12, 16]
	Main(path,ker_size_list)
