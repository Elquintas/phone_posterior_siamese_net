import os
import sys
import classes
import hparams
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
from torch.autograd import Variable
torch.manual_seed(42)



def load_files(arg1,arg2):
    
    phoneme=np.loadtxt(arg1,delimiter=',',dtype=np.float32)
    
    ref = pd.read_csv(arg2)
    ref = ref.values[:,0]
    final = []
    for i in range(len(ref)):
        #data = np.loadtxt('./data_new/'+ref[i],delimiter=',',dtype=np.float32)
        data = np.loadtxt('./data_mix/'+ref[i],delimiter=',',dtype=np.float32)
        final.append(data)

    return phoneme, final


def compare(test_phone, ref_list, model):

    set_sigmoid = []

    for i in range(len(ref_list)):

        # PAIR-WISE COMPARISON
        #x1 = test_phone.reshape(1,np.shape(test_phone)[0],np.shape(test_phone)[1])
        
        x1 = test_phone
        x2 = ref_list[i].reshape(1,np.shape(ref_list[i])[0],np.shape(ref_list[i])[1])

        len_x2 = [np.shape(ref_list[i])[0]]
        
        x2 = torch.nn.utils.rnn.pack_padded_sequence(x2,len_x2,batch_first=True)
        
        
        
        x1 = x1.cuda()
        x2 = x2.cuda()

        model.zero_grad() 
        h = model.init_hidden(1)
        h1 = h.data
        h2 = h.data

        h1 = h1.cuda()
        h2 = h2.cuda()
        
        
        #out,h1,h2,res = model(x1.to(device).float(),x2.to(device).float(),h1,h2)
        out,h1,h2,res = model(x2.to(device).float(),x1.to(device).float(),h1,h2)
        #out = res.cpu().detach().numpy()
        out = out.cpu().detach().numpy()
        
        set_sigmoid.append(out)

    
    if np.mean(set_sigmoid) > hparams.sigmoid_threshold:
        #print('negative, sim: {}'.format(np.mean(set_sigmoid)))
        return 0, len(ref_list), np.mean(set_sigmoid)
    if np.mean(set_sigmoid) < hparams.sigmoid_threshold:
        #print('positive, sim: {}'.format(np.mean(set_sigmoid)))
        return 1, len(ref_list), np.mean(set_sigmoid)



def return_batch_read(batches):
    final = []
    for i in range(len(batches)):
        data = np.loadtxt('./data/ctrls/'+batches[i],delimiter=",",dtype=np.float32)
        final.append(data)
    return final 




if __name__ == "__main__":

    device = torch.device('cuda')

    if (len(sys.argv) !=3):
        print('Usage: <Phoneme to be compared> <Reference list for phoneme>')

    # LOADING PRE-TRAINED MODEL
    model = classes.GRU_NET_DUAL(n_gru_layers=hparams.n_gru_layers,hidden_dim=hparams.hidden_dim,batch_size=1)
    model.cuda()
    checkpoint = torch.load('./checkpoints/model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()

    path = os.getcwd()
    test_phones_dir = os.path.join(path,sys.argv[1])

    acc = 0
    nr_files = 0
    avg_sim_all=[]
    for filename in os.listdir(test_phones_dir):
        if filename.endswith(".csv"):
            nr_files = nr_files + 1
            
            test_phone, ref_list = load_files(test_phones_dir+filename,sys.argv[2])
            
            #print(filename,np.shape(test_phone))
            
            for i in range(len(ref_list)):
                ref_list[i] = torch.Tensor(ref_list[i]) 
            ref_list = torch.nn.utils.rnn.pad_sequence(ref_list, batch_first=True)
         
            
            test_phone = test_phone.reshape(1,np.shape(test_phone)[0],np.shape(test_phone)[1])
            
            test_phone = torch.Tensor(test_phone)
            
            
            test_phone = torch.nn.utils.rnn.pad_sequence(test_phone, batch_first=True)
            lens = [np.shape(test_phone)[1]]
            
            test_phone = torch.nn.utils.rnn.pack_padded_sequence(test_phone,lens,batch_first=True)        
        
            res, length, out = compare(test_phone, ref_list, model)
            
            
            avg_sim_all.append(out)
            acc = acc + res
            continue

    #print(length)
    #print(nr_files)
    #print(np.shape(avg_sim_all))
    print("average sim: {}".format(np.mean(avg_sim_all)))
    print((acc/nr_files)*100)


