# Source code guideline

## Pre-set:
Download the dataset from "ftp://ftp.cbi.pku.edu.cn/pub/supplementary_file/VCNN_NIPS18/simu_data.tar.gz".
Uncompress the "simu_data_NIPS.tar.gz" and put the data directory at the root directory of the program to set up simulation dataset.

Download the parameters of the models form "ftp://ftp.cbi.pku.edu.cn/pub/supplementary_file/VCNN_NIPS18/model_parameter.tar.gz".
Uncompress the "model_parameter.tar.gz " and put the model_parameter directory at the result directory to set up parameters of Models.

Download and preprocessing Deepbind's data set. Run "python wgetall.py" under /code dir. It will download all the dataset automatically. The result will be saved in /code/cnn.csail.mit.edu/motif_discovery/.  Then Run "python seq_to_matrix.py" under /code/ dir, it will convert the original dataset into one-hot representation which can be used for further training. The result will be saved in /code/cnn.csail.mit.edu/HDF5/ . This path need to pass through command line when training the model.

REMARK: Code needs to run on python 2

## (1) Directory organization

All the paths in code directory is relative path. And by default, the root path is "../".

Simulation data is under: root + "data/". There are three simulation data sets: "simu_01", "simu_02", "simu_03" under this directory.  

Motifs in simulation dataset are saved in: root + "mtf_pwm/" and the seqLogo in under: root + "mtf_img/". Also motifs are sorted according to simulation dataset under: root + "simu_mtf/"

All results are saved under: result_root =  root + "result/". Under this directory, including following directories:

| Directory path                   | explanation                              |
| -------------------------------- | ---------------------------------------- |
| result_root + "simu_  result/"   | Save the simulation result. Each result can be find via : result_root + "simu_result/" + data_info + mode. Where data info can be "simu_01/", "simu_02/", "simu_03/". And mode can be "CNN" or "vCNN_IC". Under each folder, *.checkpointer.hdf5  file saves the model parameters, and *.pkl file save the model training results: AUC and loss |
| result_root + "raw_result/"      | The path to save theoretical scoring result |
| result_root + "simu_kerLen_auc/" | The average AUC for different kernel lengths, sorted according to different simulation dataset |
| result_root + "explain_img/"     | Save the theoretical score for different kernel and different kernels. |
| result_root + "auc_box/"         | Save the AUC box plots for different simulation datasets |
| result_root + "simu_kerDB/"      | Save the PWM recovered from kernel, via deepbind's  method |
| result_root + "mtf_ker_img/"     | save the similarity score between kernel and motif. (convolution score) |
| result_root +  "MIT_result/"     | Trained model on deepbind dataset.       |
| result_root +  "MIT_sorted/"     | Save the deepbind's AUC result and the comparation visualization results |

## (1) Theoretical scoring

Code for this section is in sorted_explain.py file. Using "ipython sorted_explain.py [parameter]" to regenerate the result.

Set [parameter] as "run_preal_plot"  to generate the theoretical score. 

Set [parameter] as " draw_detail_CNN_kerLen" to draw average AUCs for different kernel lengths and data sets.

Set [parameter] as  "visu_detail_AvePreal" to further visualize the p_real. 

### requirement:

(a) Having "simuMtf_Len-8_totIC-10.txt", "simuMtf_Len-23_totIC-12.txt" saving motif's PWM

(b) Having simulation results saved in: result_root + "simu_result/" .  Also, the result have to be organized as the one mentioned in the table above.

## (2) Test on simulation data set

### (a) Training:

Go to code directory and run: "ipython train.py  simu". Pass the key word "simu", to refer to run simulation dataset. The result will be saved in:  result_root + "simu_result/". Remark: train.py will force to Using python interpreter in PATH. And it will use "KERAS_BACKEND" by default. Run all the simulation results may take about 2 days. All the trained result and model parameters can be downloaded from "ftp://ftp.cbi.pku.edu.cn/pub/supplementary_file/VCNN_NIPS18/model_parameter.tar.gz". One can choose to download and unzip under /result dir and continue the following tests.

### (b) Draw AUC box-plot

Go to code directory and run: "ipython check_result.py simu_auc_box". The result will be saved in: result_root + "auc_box/"

### (c) Recover kernel to PWM

Using deepbind's method to generate PWM from kernel.  Go to code directory and run "ipython recover_kernel_DB.py simu_recover". The recovered PWM will be saved under root directory:  result_root+"simu_kerDB/"

### (d) Compare the similarity between kernel and motif

Using convolutional score to compare the similarity between kernel (recovered PWM) and motif.  Result saved in:  result_root + "mtf_ker_img/". Go to code directory and run "ipython check_result.py score_ker_mtf"

## (3) Compare with Deepbind's result

### (a) Requirement

Deepbind's result has been downloaded and saved as  "result/MIT_sorted/MIT_baseline.pkl" . 

### (b) Run model on Deepbind's dataset

Go to code directory and run "ipython train.py deepbind [dataDir]". The result will be saved in: result_root +  "MIT_result/". Required to download Deepbind's dataset and pass the path through the command line.  Details about dataset preparation, referring to "Pre-set" section. Run all the results may take days. One can also continue the following process by downloading the trained parameters from "ftp://ftp.cbi.pku.edu.cn/pub/supplementary_file/VCNN_NIPS18/model_parameter.tar.gz". Unzip under /result dir and continue the following tests to compare the results.

### (c) Visualize the result:

Go to code directory and run "ipython deepbind_compare.py visualize". The result will be saved under: Save it under: result_root + "MIT_sorted/".

### (d) Save AUC in csv files:

Go to code directory and run "ipython deepbind_cmpare.py save_AUC". The result will be saved under: Save it under: result_root + "MIT_sorted/". The AUC will be saved in file "AUC.csv". First column is the name of dataset, second column is VCNN's AUC, third column is Deepbind's AUC. The result is sorted by the improvement of VCNN's comparing with the one in Deepbind.
