'''
the help function commonly used
'''
import os
import h5py

def mkdir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return(True)
    else :
        return(False)
# make sure the dir is end with "/"
def check_dir_end(dir):
    dir = dir.replace("//","/")
    if not dir[-1] == "/":
        return dir +"/"
    else:
        return dir

def load_dataset(dataset):
    data = h5py.File(dataset, 'r')
    sequence_code = data['sequences'].value
    label = data['labs'].value
    return ([sequence_code, label])