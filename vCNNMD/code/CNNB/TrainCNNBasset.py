import tensorflow as tf
import os
from multiprocessing import Pool
import glob


def get_session(gpu_fraction=0.5):
	num_threads = os.environ.get('OMP_NUM_THREADS')
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
	if num_threads:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
	else:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def main(fileName):
	"""
	调用CNNB
	:return:
	"""
	pythonPath =  "/home/lijy/anaconda2/bin/ipython CNNBasset.py"

	cmd = pythonPath + " " + str(fileName)
	os.system(cmd)


if __name__ == '__main__':
	# get_session(0.7)
	DataRoot = "../../chip-seqFa/"
	
	CTCFfiles = glob.glob(DataRoot + "*Ctcf*")
	
	pool = Pool(processes=5)
	
	step = int(len(CTCFfiles)/5)
	for i in range(5):
		fileName = CTCFfiles[i * step]
		for tmp in range(i * step + 1, min((i+1)*step, len(CTCFfiles))):
			fileName = fileName + "_" + CTCFfiles[tmp]
		pool.apply_async(main, (fileName,))
	
	pool.close()
	pool.join()