'''
this file analysis the initial distribution of IC
in order to choose a good IC threshold
'''
import numpy as np
import math
from matplotlib import pyplot as plt
np.random.seed(233)

# given a column (un-normalized) using 10 as base to normalize and return the IC
def call_Entropy(col,base=10):
    col = np.array(col)
    col = np.power(base,col)
    col = col/col.sum()
    return np.array([-p*math.log(p,2.) for p in col if not p==0.]).sum()

# generate column from U(init_min,init_max) and return the IC
def get_an_IC(init_min=-0.05,init_max=0.05):
    non_IC_array = np.ones(4) * 0.25
    non_IC_entropy = call_Entropy(non_IC_array)
    tmp = np.random.random(4)*0.1-0.05
    r = non_IC_entropy - call_Entropy(tmp)
    return r

# generate hist plot of IC initial distribution
def generate_dist_histplot(output_path,sample_number = 50000):
    ret = [get_an_IC() for it in range(sample_number)]
    plt.title("IC distribution\n U[-0.05,0.05]  base = 10")
    plt.ylabel("counts")
    plt.xlabel("IC")
    plt.hist(ret)
    plt.savefig(output_path+"IC_dist.png")
    # plt.show()
    print(ret)