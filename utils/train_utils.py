import torch
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from .one_hot_dict import *
def devs_to_adj(pos_mat,n_path):
    path_node = [0]*n_path
    edges = []
    for i in range(1,len(pos_mat)):
        for j in range(n_path):
            if pos_mat[i][j] == 1:
                edges.append((int(path_node[j]),i))
                edges.append((i, int(path_node[j])))
                path_node[j] = i
    edges = list(set(edges))
    
    return [ list(x) for x in edges]




def get_loader_qfm_var_XX(data_path,n_train):
    '''
    unfixed length without padding
    '''

    with open(data_path) as f:
        n_data, n_devs, n_path, max_len = [int(x) for x in next(f).split()]
        dataset = [[float(x) for x in line.split()] for line in f]

    I = np.array(
        [[1,0],
         [0,1]],dtype=complex
    )

    PauliX = np.array(
        [[0,1],
         [1,0]],dtype=complex
    )

    hs = [
        [PauliX,PauliX,I,I],
        [PauliX,I,PauliX,I],
        [PauliX,I,I,PauliX],
        [I,PauliX,PauliX,I],
        [I,PauliX,I,PauliX],
        [I,I,PauliX,PauliX]
    ]

    H = 0
    for h in hs:
        temp_H = 1
        for i in range(4):
            temp_H = np.kron(temp_H,h[i])
        H = H + temp_H/2

    H_squre = H@H
    

    n_devs = int(len(device_dict))
    datas = []
    for i in tqdm(range(n_data)):
        n_dev = (len(dataset[i])-int(2*2**n_path))/4
        n_dev = int(n_dev)
        x_type = torch.zeros(n_dev+2,n_devs+2)
        x_pos = torch.zeros(n_dev+2,n_path)
        y_qfm = torch.zeros(1)
        #x_param = torch.zeros(max_len+2,1)
        for j in range(n_dev):
            dev = int(dataset[i][j*4])
            path1 = int(dataset[i][j*4+1])
            path2 = int(dataset[i][j*4+2]) 

            param = int(dataset[i][j*4+3])
            type_idx = one_hot(dev,param)
            x_type[j+1][type_idx+1] = 1
            x_pos[j+1][path1] = 1
            x_pos[j+1][path2] = 1

        #print(x_param)
        # start
        x_type[0][0] = 1
        x_pos[0][0:n_path] = 1
        # end
        x_type[-1][-1]=1
        x_pos[-1][0:n_path] = 1

        edges = devs_to_adj(x_pos,n_path)

        edges = torch.Tensor(edges).long().t().contiguous()
        x = torch.cat((x_type,x_pos),dim=1)
        psi_temp = dataset[i][-int(2*2**n_path):]
        psi = np.array(psi_temp[0:int(2**n_path)],dtype=complex) + 1j *np.array(psi_temp[int(2**n_path):],dtype=complex)
        temp = 4 * (psi.conj().T @ H_squre @ psi - (psi.conj().T @ H @ psi)**2)
        f_q = float(temp.real)
        y = f_q

        # choose data
        if True : 
            datas.append(Data(x=x,edge_index=edges,y=y))
    print(len(datas))
    train_dataset = datas[:n_train]
    test_dataset = datas[-5000:]

    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=32,shuffle=True)
    return train_loader, test_loader




def get_loader_qfm_var_XX_quantize2(data_path):
    '''
    unfixed length without padding
    '''
    I = np.array(
        [[1,0],
         [0,1]],dtype=complex
    )

    PauliZ = np.array(
        [[1,0],
        [0,-1]],dtype=complex
    )
    PauliX = np.array(
        [[0,1],
         [1,0]],dtype=complex
    )
    hs = [
        [PauliX,PauliX,I,I],
        [PauliX,I,PauliX,I],
        [PauliX,I,I,PauliX],
        [I,PauliX,PauliX,I],
        [I,PauliX,I,PauliX],
        [I,I,PauliX,PauliX]
    ]

    H = 0
    for h in hs:
        temp_H = 1
        for i in range(4):
            temp_H = np.kron(temp_H,h[i])
        H = H + temp_H/2

    H_squre = H@H

    with open(data_path) as f:
        n_data, n_devs, n_path, max_len = [int(x) for x in next(f).split()]
        dataset = [[float(x) for x in line.split()] for line in f]

    n_devs = int(len(device_dict_2))
    datas = []
    for i in tqdm(range(n_data)):
        n_dev = (len(dataset[i])-32)/4
        n_dev = int(n_dev)
        x_type = torch.zeros(n_dev+2,n_devs+2)
        x_pos = torch.zeros(n_dev+2,n_path)
        y_qfm = torch.zeros(1)
        #x_param = torch.zeros(max_len+2,1)
        for j in range(n_dev):
            dev = int(dataset[i][j*4])
            path1 = int(dataset[i][j*4+1])
            path2 = int(dataset[i][j*4+2]) 

            param = int(dataset[i][j*4+3])
            type_idx = one_hot2(dev,param)
            x_type[j+1][type_idx+1] = 1
            x_pos[j+1][path1] = 1
            x_pos[j+1][path2] = 1

        #print(x_param)
        # start
        x_type[0][0] = 1
        x_pos[0][0:n_path] = 1
        # end
        x_type[-1][-1]=1
        x_pos[-1][0:n_path] = 1

        edges = devs_to_adj(x_pos,n_path)

        edges = torch.Tensor(edges).long().t().contiguous()
        x = torch.cat((x_type,x_pos),dim=1)
        psi_temp = dataset[i][-32:]
        psi = np.array(psi_temp[0:16],dtype=complex) + 1j *np.array(psi_temp[16:],dtype=complex)
        temp = 4 * (psi.conj().T @ H_squre @ psi - (psi.conj().T @ H @ psi)**2)
        f_q = float(temp.real)
        y = f_q

        # choose data
        if True : 
            datas.append(Data(x=x,edge_index=edges,y=y))
    print(len(datas))
    train_dataset = datas[:50000]
    test_dataset = datas[-5000:]

    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=32,shuffle=True)
    return train_loader, test_loader




def get_loader_qfm_var_Z_quantize45(data_path,n_train):
    '''
    unfixed length without padding
    '''
    with open(data_path) as f:
        n_data, n_devs, n_path, max_len = [int(x) for x in next(f).split()]
        dataset = [[float(x) for x in line.split()] for line in f]


    I = np.array(
        [[1,0],
         [0,1]],dtype=complex
    )

    PauliZ = np.array(
        [[1,0],
        [0,-1]],dtype=complex
    )

    hs = []
    for i in range(n_path):
        hs.append([])
        for j in range(n_path):
            if j == i:
                hs[i].append(PauliZ)
            else:
                hs[i].append(I)
    H = 0
    for h in tqdm(hs):
        temp_H = 1
        for i in range(n_path):
            temp_H = np.kron(temp_H,h[i])
        H = H + temp_H/2
    H_squre = H@H
    


    n_devs = int(len(device_dict))
    datas = []
    for i in tqdm(range(n_data)):
        n_dev = (len(dataset[i])-int(2*2**n_path))/4
        n_dev = int(n_dev)
        x_type = torch.zeros(n_dev+2,n_devs+2)
        x_pos = torch.zeros(n_dev+2,n_path)
        y_qfm = torch.zeros(1)
        #x_param = torch.zeros(max_len+2,1)
        for j in range(n_dev):
            dev = int(dataset[i][j*4])
            path1 = int(dataset[i][j*4+1])
            path2 = int(dataset[i][j*4+2]) 

            param = int(dataset[i][j*4+3])
            type_idx = one_hot(dev,param)
            x_type[j+1][type_idx+1] = 1
            x_pos[j+1][path1] = 1
            x_pos[j+1][path2] = 1

        #print(x_param)
        # start
        x_type[0][0] = 1
        x_pos[0][0:n_path] = 1
        # end
        x_type[-1][-1]=1
        x_pos[-1][0:n_path] = 1

        edges = devs_to_adj(x_pos,n_path)

        edges = torch.Tensor(edges).long().t().contiguous()
        x = torch.cat((x_type,x_pos),dim=1)
        psi_temp = dataset[i][-int(2*2**n_path):]
        psi = np.array(psi_temp[0:int(2**n_path)],dtype=complex) + 1j *np.array(psi_temp[int(2**n_path):],dtype=complex)
        temp = 4 * (psi.conj().T @ H_squre @ psi - (psi.conj().T @ H @ psi)**2)
        f_q = float(temp.real)
        y = f_q

        # choose data
        if True : 
            datas.append(Data(x=x,edge_index=edges,y=y))
    print(len(datas))
    train_dataset = datas[:n_train]
    test_dataset = datas[-5000:]

    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=32,shuffle=True)
    return train_loader, test_loader

def get_loader_qfm_var_X_quantize45(data_path,n_train):
    '''
    unfixed length without padding
    '''
    with open(data_path) as f:
        n_data, n_devs, n_path, max_len = [int(x) for x in next(f).split()]
        dataset = [[float(x) for x in line.split()] for line in f]


    I = np.array(
        [[1,0],
         [0,1]],dtype=complex
    )

    PauliX = np.array(
        [[0,1],
         [1,0]],dtype=complex
    )


    hs = []
    for i in range(n_path):
        hs.append([])
        for j in range(n_path):
            if j == i:
                hs[i].append(PauliX)
            else:
                hs[i].append(I)
    H = 0
    for h in tqdm(hs):
        temp_H = 1
        for i in range(n_path):
            temp_H = np.kron(temp_H,h[i])
        H = H + temp_H/2
    H_squre = H@H
    


    n_devs = int(len(device_dict))
    datas = []
    for i in tqdm(range(n_data)):
        n_dev = (len(dataset[i])-int(2*2**n_path))/4
        n_dev = int(n_dev)
        x_type = torch.zeros(n_dev+2,n_devs+2)
        x_pos = torch.zeros(n_dev+2,n_path)
        y_qfm = torch.zeros(1)
        #x_param = torch.zeros(max_len+2,1)
        for j in range(n_dev):
            dev = int(dataset[i][j*4])
            path1 = int(dataset[i][j*4+1])
            path2 = int(dataset[i][j*4+2]) 

            param = int(dataset[i][j*4+3])
            type_idx = one_hot(dev,param)
            x_type[j+1][type_idx+1] = 1
            x_pos[j+1][path1] = 1
            x_pos[j+1][path2] = 1

        #print(x_param)
        # start
        x_type[0][0] = 1
        x_pos[0][0:n_path] = 1
        # end
        x_type[-1][-1]=1
        x_pos[-1][0:n_path] = 1

        edges = devs_to_adj(x_pos,n_path)

        edges = torch.Tensor(edges).long().t().contiguous()
        x = torch.cat((x_type,x_pos),dim=1)
        psi_temp = dataset[i][-int(2*2**n_path):]
        psi = np.array(psi_temp[0:int(2**n_path)],dtype=complex) + 1j *np.array(psi_temp[int(2**n_path):],dtype=complex)
        temp = 4 * (psi.conj().T @ H_squre @ psi - (psi.conj().T @ H @ psi)**2)
        f_q = float(temp.real)
        y = f_q

        # choose data
        if True : 
            datas.append(Data(x=x,edge_index=edges,y=y))
    print(len(datas))
    train_dataset = datas[:n_train]
    test_dataset = datas[-5000:]

    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=32,shuffle=True)
    return train_loader, test_loader

def get_loader_qfm_var_Y_quantize45(data_path,n_train):
    '''
    unfixed length without padding
    '''
    with open(data_path) as f:
        n_data, n_devs, n_path, max_len = [int(x) for x in next(f).split()]
        dataset = [[float(x) for x in line.split()] for line in f]


    I = np.array(
        [[1,0],
         [0,1]],dtype=complex
    )

    PauliY = np.array(
        [[0,-1j],
        [1j,0]],dtype=complex
    )


    hs = []
    for i in range(n_path):
        hs.append([])
        for j in range(n_path):
            if j == i:
                hs[i].append(PauliY)
            else:
                hs[i].append(I)
    H = 0
    for h in tqdm(hs):
        temp_H = 1
        for i in range(n_path):
            temp_H = np.kron(temp_H,h[i])
        H = H + temp_H/2
    H_squre = H@H
    


    n_devs = int(len(device_dict))
    datas = []
    for i in tqdm(range(n_data)):
        n_dev = (len(dataset[i])-int(2*2**n_path))/4
        n_dev = int(n_dev)
        x_type = torch.zeros(n_dev+2,n_devs+2)
        x_pos = torch.zeros(n_dev+2,n_path)
        y_qfm = torch.zeros(1)
        #x_param = torch.zeros(max_len+2,1)
        for j in range(n_dev):
            dev = int(dataset[i][j*4])
            path1 = int(dataset[i][j*4+1])
            path2 = int(dataset[i][j*4+2]) 

            param = int(dataset[i][j*4+3])
            type_idx = one_hot(dev,param)
            x_type[j+1][type_idx+1] = 1
            x_pos[j+1][path1] = 1
            x_pos[j+1][path2] = 1

        #print(x_param)
        # start
        x_type[0][0] = 1
        x_pos[0][0:n_path] = 1
        # end
        x_type[-1][-1]=1
        x_pos[-1][0:n_path] = 1

        edges = devs_to_adj(x_pos,n_path)

        edges = torch.Tensor(edges).long().t().contiguous()
        x = torch.cat((x_type,x_pos),dim=1)
        psi_temp = dataset[i][-int(2*2**n_path):]
        psi = np.array(psi_temp[0:int(2**n_path)],dtype=complex) + 1j *np.array(psi_temp[int(2**n_path):],dtype=complex)
        temp = 4 * (psi.conj().T @ H_squre @ psi - (psi.conj().T @ H @ psi)**2)
        f_q = float(temp.real)
        y = f_q

        # choose data
        if True : 
            datas.append(Data(x=x,edge_index=edges,y=y))
    print(len(datas))
    train_dataset = datas[:n_train]
    test_dataset = datas[-5000:]

    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=32,shuffle=True)
    return train_loader, test_loader

def get_loader_qfm_var_quantize2(data_path,n_train,pauli):
    '''
    unfixed length without padding
    '''
    with open(data_path) as f:
        n_data, n_devs, n_path, max_len = [int(x) for x in next(f).split()]
        dataset = [[float(x) for x in line.split()] for line in f]


    I = np.array(
        [[1,0],
         [0,1]],dtype=complex
    )

    Pauli = {
        'X': np.array(
            [[0,1],
            [1,0]],dtype=complex
        ),
        'Y': np.array(
            [[0,-1j],
            [1j,0]],dtype=complex
        ),
        'Z': np.array(
            [[1,0],
        [0,-1]],dtype=complex
        )
    }


    hs = []
    for i in range(n_path):
        hs.append([])
        for j in range(n_path):
            if j == i:
                hs[i].append(Pauli[pauli])
            else:
                hs[i].append(I)
    H = 0
    for h in tqdm(hs):
        temp_H = 1
        for i in range(n_path):
            temp_H = np.kron(temp_H,h[i])
        H = H + temp_H/2
    H_squre = H@H
    


    n_devs = int(len(device_dict_2))
    datas = []
    for i in tqdm(range(n_data)):
        n_dev = (len(dataset[i])-int(2*2**n_path))/4
        n_dev = int(n_dev)
        x_type = torch.zeros(n_dev+2,n_devs+2)
        x_pos = torch.zeros(n_dev+2,n_path)
        y_qfm = torch.zeros(1)
        #x_param = torch.zeros(max_len+2,1)
        for j in range(n_dev):
            dev = int(dataset[i][j*4])
            path1 = int(dataset[i][j*4+1])
            path2 = int(dataset[i][j*4+2]) 

            param = int(dataset[i][j*4+3])
            type_idx = one_hot2(dev,param)
            x_type[j+1][type_idx+1] = 1
            x_pos[j+1][path1] = 1
            x_pos[j+1][path2] = 1

        #print(x_param)
        # start
        x_type[0][0] = 1
        x_pos[0][0:n_path] = 1
        # end
        x_type[-1][-1]=1
        x_pos[-1][0:n_path] = 1

        edges = devs_to_adj(x_pos,n_path)

        edges = torch.Tensor(edges).long().t().contiguous()
        x = torch.cat((x_type,x_pos),dim=1)
        psi_temp = dataset[i][-int(2*2**n_path):]
        psi = np.array(psi_temp[0:int(2**n_path)],dtype=complex) + 1j *np.array(psi_temp[int(2**n_path):],dtype=complex)
        temp = 4 * (psi.conj().T @ H_squre @ psi - (psi.conj().T @ H @ psi)**2)
        f_q = float(temp.real)
        y = f_q

        # choose data
        if True : 
            datas.append(Data(x=x,edge_index=edges,y=y))
    print(len(datas))
    train_dataset = datas[:n_train]
    test_dataset = datas[60000:]

    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=32,shuffle=True)
    return train_loader, test_loader

def predictor_loss(y_hat, y_target):
    loss = F.mse_loss(y_hat,y_target)
    return loss