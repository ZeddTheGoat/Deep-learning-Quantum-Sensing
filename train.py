import torch
import torch.nn as nn
#from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import autograd
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from models.model import GNN_QFM, MLP
from tqdm import tqdm
import math
from utils.one_hot_dict import *
from utils.train_utils import get_loader_qfm_var_XX, predictor_loss, get_loader_qfm_var_Z_quantize45, get_loader_qfm_var_X_quantize45, get_loader_qfm_var_Y_quantize45, get_loader_qfm_var_quantize2
import numpy as np
import numpy as np
import mlflow.pytorch

device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'

args = {
    "n_path":4,
    'device_dict':device_dict,
    'n_train':3000,
    'pauli': 'Z',
    'quantize': 45
}

feature_size = len(args['device_dict']) + 2 + args['n_path'] # tpyes + start + end +  position


model = GNN_QFM(feature_size=feature_size,seq_size=19,output_size=1,hidden_size=256,num_layers=4,dropout=0)
model = model.to(device)

loss_fn = predictor_loss
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)

best_score = 15

def run_one_epoch(data_loader:DataLoader, type, epoch):
    all_losses = []
    global best_score
    for _, batch in enumerate(tqdm(data_loader)):
        try:
            batch.to(device)
            optimizer.zero_grad()

            y_hat = model(
                batch.x.float(),
                batch.edge_index,
                batch.batch
            )
            
            loss= loss_fn(y_hat,batch.y)
            if type == "Train":
            
                loss.backward()  
                optimizer.step() 
            # Store loss and metrics
            all_losses.append(loss.detach().cpu().numpy())
            #all_accs.append(acc)
        except IndexError as error:
            print("Error: ", error)
    
    print(f"{type} epoch {epoch} loss: ", np.array(all_losses).mean())
    mlflow.log_metric(key=f"{type} Epoch Loss", value=float(np.array(all_losses).mean()), step=epoch)

    if type == 'Test' and np.array(all_losses).mean() < best_score:
        best_score = np.array(all_losses).mean()
        torch.save(model.state_dict(),'./weights/predictor_{}_{}qubit_quantize{}_{}.pt'.format(args['pauli'],args['n_path'],args['quantize'],args['n_train']))

data_path = './dataset/dataset_{}qubit_quantize{}.txt'.format(args['n_path'],args['quantize'])
train_loader, test_loader = get_loader_qfm_var_Z_quantize45(data_path,args['n_train'])
num_epochs = 200
mlflow.set_tracking_uri('./mlrun')
with mlflow.start_run(run_name='test') as run:
    for epoch in range(num_epochs):
        model.train()
        run_one_epoch(train_loader,type='Train', epoch=epoch)
        if epoch%5 == 0:
            print("Start test epoch...")
            model.eval()
            run_one_epoch(test_loader, type='Test', epoch=epoch)

