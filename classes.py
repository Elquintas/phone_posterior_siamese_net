import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hparams
import sys
import pandas as pd
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch.nn import Embedding
torch.manual_seed(42)

device = torch.device("cpu")



class GRU_NET_DUAL(nn.Module):
    def __init__(self, n_gru_layers, hidden_dim, batch_size):
        super(GRU_NET_DUAL, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_gru_layers = n_gru_layers
        self.batch_size = batch_size
    
        # removed dropout from the GRU layers
        self.gru = nn.GRU(hparams.feats_dim ,self.hidden_dim, self.n_gru_layers, batch_first=True, dropout=hparams.rnn_dropout,
                bidirectional=hparams.bidirectional) #previously input = 54
        
        self.fc1 = nn.Linear(self.hidden_dim*2,self.hidden_dim*2)
        self.fc2 = nn.Linear(self.hidden_dim*2,self.hidden_dim*2)
        self.fc3 = nn.Linear(self.hidden_dim*2,1)
        #self.fc4 = nn.Linear(self.hidden_dim,1)

        self.fc1_siam = nn.Linear(54,64)
        self.fc2_siam = nn.Linear(64,32)
 
        self.bn1 = nn.BatchNorm1d(num_features=self.hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(num_features=self.hidden_dim*2)
        self.bn3 = nn.BatchNorm1d(num_features=self.hidden_dim*2)
        #self.bn4 = nn.BatchNorm1d(num_features=self.hidden_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=hparams.dropout)
        self.sigmoid = nn.Sigmoid()


    def forward(self,input1,input2,h1,h2):
        
        output1, h1 = self.gru(input1,h1)
        output2, h2 = self.gru(input2,h2)
        
        #hidden1 = h1[2]
        #hidden2 = h2[2]
        

        res = 0

        # UNPACKING
        # PICK THE LAST HIDDEN STATE FROM EACH SEQUENCE
        output1, i1 = torch.nn.utils.rnn.pad_packed_sequence(output1)
        output2, i2 = torch.nn.utils.rnn.pad_packed_sequence(output2)
       
        #out1 = torch.Tensor(np.zeros((self.batch_size,self.hidden_dim)))
        #out2 = torch.Tensor(np.zeros((self.batch_size,self.hidden_dim)))
        
        #out_bi_1 = torch.Tensor(np.zeros((self.batch_size,self.hidden_dim)))
        #out_bi_2 = torch.Tensor(np.zeros((self.batch_size,self.hidden_dim)))

        #for i in range(len(i1)):
        #    out1[i]=output1[i1[i]-1,i]
        #    out2[i]=output2[i2[i]-1,i]
            
        #    out_bi_1[i]=output1[0,i]
        #    out_bi_2[i]=output2[0,i]
        
        #o1 = torch.cat((out1,out_bi_1),dim=1)
        #o2 = torch.cat((out1,out_bi_2),dim=1)
        
        h1_fin = torch.cat((h1[3],h1[2]),dim=1)
        h2_fin = torch.cat((h2[3],h2[2]),dim=1)

        #alal = torch.abs(out1-out2)
        alal = torch.abs(h1_fin-h2_fin)        
        
        #x = torch.abs(output1[:,-1]-output2[:,-1])
        #x = torch.abs(output1[-1]-output2[-1])
        
        #res = F.pairwise_distance(out1,out2)
        #res = torch.mul(res,res)
        #res = torch.sum(res,axis=1)
        #res = torch.sqrt(res)

        #experiment with embeddings from hidden state
        #x = torch.abs(hidden1-hidden2)        

        x = self.bn1(alal.cuda())
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.bn3(x)
        x = self.dropout(x)
        x = self.fc3(x)
        

        #x = self.bn3(x)
        #x = self.fc3(x)
        out = self.sigmoid(x)
        #x = self.bn2(x)
        
        
        #x = self.dropout(x)
        #out = self.fc2(x)
        #x = self.relu(x)
        
        #x = self.bn3(x)
        #out = self.fc3(x)

        #out = self.fc2(x)
        #out = self.sigmoid(out)
        
        return out, h1, h2, res

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        # multiply n_gru_layers by 2 if bidirectional gru is being used
        hidden = weight.new(self.n_gru_layers*2,self.batch_size, self.hidden_dim).zero_().to(device)
        return hidden



def read_file(filename):
    data = np.loadtxt(filename,delimiter=',',dtype=np.float32)
    return data



class sequential_train(Dataset):
    def __init__(self,transform=None):
        #xy = pd.read_csv('./FINAL_NEGPOS_TRAIN.csv')
        xy = pd.read_csv('./individual_phonemes/f_test_drive.csv')
        #print(xy.values[:,0])
        
        self.y1 = np.asarray(xy.values[:,1])
        self.y2 = np.asarray(xy.values[:,3])

        self.x1 = xy.values[:,0]
        self.x2 = xy.values[:,2]
        
        self.n_samples = xy.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x1[idx],self.y1[idx],self.x2[idx],self.y2[idx]


class sequential_validation(Dataset):
    def __init__(self,transform=None):
        #xy = pd.read_csv('./FINAL_NEGPOS_VALIDATION.csv')
        xy = pd.read_csv('./VALIDATION_16_SEL.csv')
        #print(xy.values[:,0])

        self.y1 = np.asarray(xy.values[:,1])
        self.y2 = np.asarray(xy.values[:,3])

        self.x1 = xy.values[:,0]
        self.x2 = xy.values[:,2]
                
        self.n_samples = xy.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x1[idx],self.y1[idx],self.x2[idx],self.y2[idx]

class sequential_test(Dataset):
    def __init__(self,transform=None):
        #xy = pd.read_csv('./FINAL_NEGPOS_TEST.csv')
        xy = pd.read_csv('./TEST_16_SEL.csv')
        #print(xy.values[:,0])

        self.y1 = np.asarray(xy.values[:,1])
        self.y2 = np.asarray(xy.values[:,3])

        self.x1 = xy.values[:,0]
        self.x2 = xy.values[:,2]

        self.n_samples = xy.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x1[idx],self.y1[idx],self.x2[idx],self.y2[idx]



