#import torch
#device_dict = [
#    (1,0,1),(1,0,2),(1,0,3),(1,1,0),(1,1,2),(1,1,3),(1,2,0),(1,2,1),(1,2,3),(1,3,0),(1,3,1),(1,3,2),
#    (2,0,1),(2,0,2),(2,0,3),(2,1,0),(2,1,2),(2,1,3),(2,2,0),(2,2,1),(2,2,3),(2,3,0),(2,3,1),(2,3,2),
#    (3,0,0),(3,1,1),(3,2,2),(3,3,3),
#    (4,0,0),(4,1,1),(4,2,2),(4,3,3),
#    (5,0,0),(5,1,1),(5,2,2),(5,3,3),
#    (6,0,1),(6,2,3),
#    (7,0,1),(7,2,3),
#    (8,0,1),(8,2,3),
#]

device_dict = [
    (1,0),(2,0),(3,1),(3,2),(3,3),(3,4),(4,1),(4,2),(4,3),(4,4), (5,0),(6,0),(7,0),(8,0)
]

device_dict_2 = [
    (1,0),(2,0), (5,0),(6,0),(7,0),(8,0)
] + [(3,int(i)) for i in range(1,91)]+ [(4,int(i)) for i in range(1,91)]

def one_hot(device, param):
    return device_dict.index((device,param))

def one_hot2(device, param):
    return device_dict_2.index((device,param))

def onehot_to_idx(one_hot):
    return one_hot.tolist().index(1)