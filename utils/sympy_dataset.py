import sympy as sp
import numpy as np
import math
from numpy import linalg as LA
from device import *

def TermsCoeffList(expr, match_factor): # retutn [(terms, coeff),(terms, coeff),...]    
    dictadd=sp.collect(expr, [match_factor], evaluate=False)
    TermsCoeff=list(dictadd.items())
    return TermsCoeff
def replaceRule(expr, repls):
    for k, m in repls.items():
        expr = expr.replace(k, m, map=False, simultaneous=True, exact=False)   
    return expr

def StateVector(terms,n_path):
    if len(terms) == 0:
        psi = np.zeros([2]*n_path,dtype=complex).reshape(-1)
        return psi
    psi = np.zeros([2]*n_path,dtype=complex)
    for k, v in terms:
        idx = [k.args[i].indices[0] for i in range(4) ]
        amp = complex(v)
        psi[tuple(idx)]=amp
    psi = psi.reshape(-1)
    psi = psi/LA.norm(psi)
    return psi

def action(act_id:int,**kwargs):

    # NULL
    if act_id == 0:
        return kwargs['expr']

    # Beam Splitter
    if act_id == 1:
        return BS(kwargs['expr'],kwargs['p1'],kwargs['p2'])
    
    # 
    if act_id == 2:
        return PBS(kwargs['expr'], kwargs['p1'], kwargs['p2'])
    
    if act_id == 3:
        return HWP(kwargs['expr'], kwargs['p1'], kwargs['theta'])

    if act_id == 4:
        return QWP(kwargs['expr'], kwargs['p1'], kwargs['theta'])

    if act_id == 5:
        return R(kwargs['expr'], kwargs['p1'])

def apply_actions(expr, act_list):
    N = len(act_list)
    expr_temp = expr
    for i in range(N):
        act_id = act_list[i][0]
        p1 = act_list[i][1]
        p2 = act_list[i][2]
        theta = act_list[i][3]
        expr_temp = action(act_id=act_id, expr=expr_temp,p1=p1,p2=p2,theta=theta)
    return expr_temp

def print_action(act_id:int,**kwargs):
    if act_id == 1:
        print('Beam Splitter on path {} and path {}'.format(kwargs['p1'],kwargs['p2'])) 
    
    # 
    if act_id == 2:
        print('Polarized Beam Splitter on path {} and path {}'.format(kwargs['p1'],kwargs['p2'])) 
        
    
    if act_id == 3:
        print('Half-Wave Plate on path {} with theta {}'.format(kwargs['p1'], kwargs['theta'])) 

    if act_id == 4:
        print('Quater-Wave Plate on path {} with theta {}'.format(kwargs['p1'], kwargs['theta'])) 

    if act_id == 5:
        print('Reflection on path {}'.format(kwargs['p1']))

def print_act_list(act_list):
    N = len(act_list)
    for i in range(N):
        act_id = act_list[i][0]
        p1 = act_list[i][1]
        p2 = act_list[i][2]
        theta = act_list[i][3]
        print_action(act_id=act_id,p1=p1,p2=p2,theta=theta)

def generate_random_action(n_path):
    act_id = int(np.random.randint(low=1,high=6))

    if act_id == 1 or act_id == 2:
        p1, p2 = np.random.default_rng().choice(n_path, 2,replace=False)
        act_vec = [act_id,int(p1),int(p2),0]
        return act_vec

    if act_id == 3 or act_id == 4:
        p1 = int(np.random.randint(n_path))
        theta = float( np.random.uniform(0,2*math.pi))
        act_vec = [act_id, p1, p1, theta]
        return act_vec

    if act_id == 5:
        p1 = int(np.random.randint(n_path))
        act_vec = [act_id, p1, p1, 0]
        return act_vec

def act_to_vec(act_name:str, p1, p2, theta):
    act_dict = {'NULL':0,'BS':1,'PBS':2,'HWP':3,'QWP':4,'R':5}
    act_vec = [act_dict[act_name], p1, p2, theta]
    return act_vec
