import os
import glob


CTCFfiles = glob.glob("../../Peak/*Ctcf*")
for file in CTCFfiles:
	filename = file.split("/")[-1].replace("narrowPeak", "")
	cmd = "sbatch Compare.sh " + filename
	print(cmd)
	os.system(cmd)
