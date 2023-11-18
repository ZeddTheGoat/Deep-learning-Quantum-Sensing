from device import *
from utils import *
import math
import sympy as sp
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path',type=str)
args=parser.parse_args()
data_path = args.path

a, b, c, d, e, f, FF1, FF2, FF3, FF4, FFn, HH, GG1, GG2, GG3, GG4=map(sp.IndexedBase,['a','b','c','d','e', 'f', 'FF1','FF2','FF3','FF4','FFn','HH','GG1','GG2','GG3','GG4'])  
g, h, m,n,o,p,q,r = map(sp.IndexedBase, ['g','h','m','n','o','p','q','r'])
l,l1, l2, l3, l4, l5, l6, l7, l8, x1, x2, x3, x4, x5, x6, coeff, powern =map(sp.Wild,['l','l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'coeff', 'powern'])
x7, x8, x9, x10, x11, x12, x13, x14 = map(sp.Wild,[ 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14'])
path_table = {0:a, 1:b, 2:c, 3:d, 4:e, 5:f, 6:g, 7:h, 8:m, 9:n, 10:o, 11:p, 12:q, 13:r}

sqrt2 = sp.sqrt(2)

n_data = 1000
n_path = 8
n_devs = 5 + 3
n_quantization = 4
MIN_LEN = 3
MAX_LEN = 15

data_info = [n_data,n_devs,n_path,MAX_LEN]
with open(data_path,'a+') as tfile:
    tfile.write(" ".join([str(num) for num in data_info])+'\n')



rng = np.random.default_rng()
for _ in tqdm(range(n_data)):


    n_DC = int(n_path/2)
    initial_state = 1
    initial_mat = []
    for i in range(n_DC):
        act_id = int(rng.integers(low=6,high=9))
        initial_mat.append([act_id,2*i,2*i+1,0])
        initial_state = initial_state * action(act_id=act_id,p1=path_table[2*i],p2=path_table[2*i+1])
    
    #
    # print(initial_state)

    n_device = rng.integers(low = MIN_LEN, high=MAX_LEN+1)
    act_mat = [ generate_random_action(n_path,n_quantization) for _ in range(n_device) ]
    act_list = [[int(act_vec[0]), path_table[int(act_vec[1])], path_table[int(act_vec[2])], act_vec[3] * sp.pi/n_quantization  ] for act_vec in act_mat]

    acts = []
    for act in initial_mat:
        acts = acts + act
    for act in act_mat:
        acts = acts + act
    #print(acts)




    expr = apply_actions(initial_state, act_list)

    
    NFoldrepls = {coeff*a[l1]*a[l2] : 0, coeff*b[l1]*b[l2] : 0, coeff*c[l1]*c[l2] : 0, coeff*d[l1]*d[l2] : 0,coeff*e[l1]*e[l2] : 0,coeff*f[l1]*f[l2] : 0,coeff*g[l1]*g[l2] : 0, coeff*h[l1]*h[l2] : 0}
    
    expr=replaceRule(expr,NFoldrepls)

    terms = TermsCoeffList(expr, a[x1]*b[x2]*c[x3]*d[x4]*e[x5]*f[x6]*g[x7]*h[x8])

    psi = StateVector(terms,n_path)

    psi_real = [ float(amp.real) for amp in psi ]
    psi_imag = [float(amp.imag) for amp in psi]
    line_str = [str(num) for num in acts] + [str(num) for num in psi_real] + [str(num) for num in psi_imag]
    
    with open(data_path,'a+') as tfile:
        tfile.write(" ".join(line_str) + "\n")

