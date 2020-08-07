import torch
import sys
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Embedding
import torch.nn.init
import torchvision
import classes
import hparams
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
torch.manual_seed(42)

#   Pathological Speech labels = 1
#   Healthy Speech labels = 0
#
#   Final Labels:
#       1 - different pairs
#       0 - similar pairs


def contrastive_loss(label,euc_dist):
    return torch.mean((label)*torch.pow(euc_dist,2)+
                (1-label)*torch.pow(torch.clamp(1-euc_dist,min=0.0),2))


def oneshot(x1,x2,label):
    euc_dist = F.pairwise_distance(x1,x2)
    ed = euc_dist.detach().numpy()
    if euc_dist > hparams.euc_dist_threshold:
        print('Euc. Dist: {} - Different pair 1 - Target lab. - {}'.format(ed,label))
        return 1
    else:
        print('Euc. Dist: {} -   Same pair    0 - Target lab. - {}'.format(ed,label))
        return 0
    
def acc_calc(predicted,target):
    size = len(predicted)
    tot = np.abs(predicted-target)
    tot = np.sum(tot)

    acc = ((size-tot)/size)*100
    print('Accuracy: {}'.format(acc))

def precision_recall(predicted,target):
    
    tn=0
    tp=0
    fn=0
    fp=0

    for i in range(len(predicted)):
        if (predicted[i]==target[i]==1):
            tp = tp+1
        if (predicted[i]==target[i]==0):
            tn = tn+1
        if (target[i]==1 and predicted[i]==0):
            fn = fn+1
        if (target[i]==0 and predicted[i]==1):
            fp = fp+1
    print('Precision: {}'.format(tp/(tp+fp)))
    print('Recall: {}'.format(tp/(tp+fn)))

    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)

    print('TPR: {}'.format(tpr))
    print('FPR: {}'.format(fpr))

    auc = roc_auc_score(target,predicted)
    print('AUC: {}'.format(auc))

def plot_losses(train,validation):
    plt.title('Train and Validation Losses')
    line1, = plt.plot(train,'b',label="Train")
    line2, = plt.plot(validation,'g',label="Validation")
    plt.legend(handles=[line1,line2])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def return_batch_read(batches):
    final = []
    for i in range(len(batches)):
        data = np.loadtxt('./data_mix/'+batches[i],delimiter=",",dtype=np.float32)
        #data = data.T
        final.append(data)
    return final



if __name__ == "__main__":
    
    device = torch.device('cuda')
    
    sequential_train = classes.sequential_train
    sequential_validation = classes.sequential_validation
    sequential_test = classes.sequential_test

    #Loading Datasets
    dataset = sequential_train()
    trainloader = DataLoader(dataset=dataset, batch_size=hparams.batch_size, shuffle=True,drop_last=True) 
    dataset = sequential_validation()
    validationloader = DataLoader(dataset=dataset, batch_size=hparams.batch_size, shuffle=True,drop_last=True)
    dataset = sequential_test()
    testloader = DataLoader(dataset=dataset, batch_size=hparams.batch_size, shuffle=True,drop_last=True)
    
    model = classes.GRU_NET_DUAL(n_gru_layers=hparams.n_gru_layers,hidden_dim=hparams.hidden_dim,batch_size=hparams.batch_size)
    
    #Orthogonal initialization of weights:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
        if isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                if 'bias' in name:
                    param.data.fill_(0)
                
    
    model.cuda()
    # To Load a pre-trained model:
    #checkpoint = torch.load('./checkpoints/model.pth')

    optimizer = torch.optim.Adam(model.parameters(),lr=hparams.lr,weight_decay=0.00001) #,weight_decay=0.00001)#0.0001)
    #optimizer = torch.optim.Adadelta(model.parameters(),lr=hparams.lr)
    #optimizer = torch.optim.Adamax(model.parameters(),lr=hparams.lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=hparams.lr)   #momentum 0.9?
    
    
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #ep = checkpoint['epoch']
    #print('LOST EPOCH: {}'.format(ep)) 
    
    criterion = nn.BCELoss()
    #criterion = nn.BCEWithLogitsLoss()
    loss_log=[]
    val_loss_track=[]
    train_loss_track=[]
    train_loss=[]
    
    iteration = 1
    
    for ep in range(hparams.epochs):
                     
        model.train()
        train_loss=[]
        
        for batch_idx, (x1,y1,x2,y2) in enumerate(trainloader):
            if(batch_idx%1000 == 0):
                print(ep,batch_idx)

            final_label = np.abs(y2-y1)
            #print(batch_idx)    
            
            #print(x1)
            #print(x2)
            x1 = return_batch_read(x1)
            x2 = return_batch_read(x2)

            # invert labels (1- positive pair, 0 - negative pair)
            #final_label = final_label-1
            #final_label = np.abs(final_label)

            final_label = final_label.cuda()   
            
            final_label = final_label.reshape(hparams.batch_size,1)
            
            h = model.init_hidden(hparams.batch_size)
            
            h1 = h.data
            h2 = h.data
            h1 = h1.cuda()
            h2 = h2.cuda()

            model.zero_grad()


            # Network with fully connected layer at the end
            #output, h1, h2, euc_dist = model(x1.to(device).float(),x2.to(device).float(),h1,h2) 

            for i in range(len(x1)):                
                
                #print(np.shape(x2[i]))
                #x2[i] = x2[i].squeeze() 
                #x1[i] = x1[i].squeeze()
                #print(np.shape(x2[16]))
                
                x1[i] = torch.Tensor(x1[i])
                x2[i] = torch.Tensor(x2[i])
            

            len_x1=[]
            len_x2=[]
            for i in range(len(x1)):
                len_x1.append(len(x1[i]))
                len_x2.append(len(x2[i]))


            
            x1 = torch.nn.utils.rnn.pad_sequence(x1, batch_first=True)
            x2 = torch.nn.utils.rnn.pad_sequence(x2, batch_first=True)

            x1 = x1.reshape(hparams.batch_size,max(len_x1),hparams.feats_dim)   #previously 54
            x2 = x2.reshape(hparams.batch_size,max(len_x2),hparams.feats_dim)   #previously 54
            
            x1 = torch.nn.utils.rnn.pack_padded_sequence(x1,len_x1,batch_first=True, enforce_sorted=False)
            x2 = torch.nn.utils.rnn.pack_padded_sequence(x2,len_x2,batch_first=True, enforce_sorted=False)

            x1 = x1.cuda()
            x2 = x2.cuda()
            
            output, h1, h2, euc_dist = model(x1,x2,h1,h2)            
            
            loss = criterion(output.squeeze(),final_label.float().squeeze())
            #loss = contrastive_loss(final_label,euc_dist.cuda())
            
            train_loss.append(loss.cpu().detach().numpy())
            
            if batch_idx % 10 ==0:
                loss_log.append(loss)
            
            loss.backward()
            optimizer.step()
            

            # Checkpoint Saving
        #if (ep % hparams.checkpoint_interval == 0 and ep != 0):
        print('SAVING CHECKPOINT AT EPOCH NR: {}'.format(ep))
        torch.save({'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, "./checkpoints/model.pth")

        
        # Validation at end of each epoch
        model.eval()
        h = model.init_hidden(hparams.batch_size)
        h1 = h.data
        h2 = h.data
        validation_loss = []
        for batch_idx, (x1,y1,x2,y2) in enumerate(validationloader):
            
            x1 = return_batch_read(x1)
            x2 = return_batch_read(x2)
            
            h = model.init_hidden(hparams.batch_size)
            h1 = h.data
            h2 = h.data
            h1 = h1.cuda()
            h2 = h2.cuda()

            final_label_val = np.abs(y1-y2)
            final_label_val = final_label_val.cuda()

            for i in range(len(x1)):
                x1[i] = torch.Tensor(x1[i])
                x2[i] = torch.Tensor(x2[i])

            len_x1=[]
            len_x2=[]
            for i in range(len(x1)):
                len_x1.append(len(x1[i]))
                len_x2.append(len(x2[i]))


            x1 = torch.nn.utils.rnn.pad_sequence(x1, batch_first=True)
            x2 = torch.nn.utils.rnn.pad_sequence(x2, batch_first=True)
            x1 = x1.reshape(hparams.batch_size,max(len_x1),hparams.feats_dim)   #previously 54
            x2 = x2.reshape(hparams.batch_size,max(len_x2),hparams.feats_dim)   #previously 54
            x1 = torch.nn.utils.rnn.pack_padded_sequence(x1,len_x1,batch_first=True, enforce_sorted=False)
            x2 = torch.nn.utils.rnn.pack_padded_sequence(x2,len_x2,batch_first=True, enforce_sorted=False)
            x1 = x1.cuda()
            x2 = x2.cuda()




            # invert labels (1- positive pair, 0 - negative pair)
            #final_label_val = final_label_val-1
            #final_label_val = np.abs(final_label_val)


            #final_label_val = final_label_val.reshape(hparams.batch_size,1)
                
            val_out, h1, h2, euc_dist = model(x1,x2,h1,h2)
            
            val_loss = criterion(val_out.squeeze(),final_label_val.float().squeeze())
            #val_loss = contrastive_loss(final_label,euc_dist.cuda())

            validation_loss.append(val_loss.cpu().detach().numpy())
        
        
        
        val_loss_track.append(np.mean(validation_loss))
        train_loss_track.append(np.mean(train_loss))
        print('EPOCH: {} -- VALIDATION LOSS: {} -- TRAINING LOSS: {}'.format(ep,np.mean(validation_loss),np.mean(train_loss)))
        train_loss=[]

   


    print('\nTRAIN SET METRICS:')

    # Test phase after training
    model.eval()
    h = model.init_hidden(hparams.batch_size)
    h1 = h.data
    h2 = h.data

    test_target = []
    test_predicted = []

    for batch_idx, (x1,y1,x2,y2) in enumerate(testloader):
        h1 = h.data
        h2 = h.data
        h1 = h1.cuda()
        h2 = h2.cuda()

        final_labels = np.abs(y1-y2)
        
        # invert labels (1- positive pair, 0 - negative pair)
        #final_labels = final_labels-1
        #final_labels = np.abs(final_labels)
        
        final_labels = final_labels.cuda()

        test_target.append(final_labels.cpu().numpy())

        x1 = return_batch_read(x1)
        x2 = return_batch_read(x2)

        for i in range(len(x1)):
            x1[i] = torch.Tensor(x1[i])
            x2[i] = torch.Tensor(x2[i])

        len_x1=[]
        len_x2=[]
        for i in range(len(x1)):
            len_x1.append(len(x1[i]))
            len_x2.append(len(x2[i]))


        x1 = torch.nn.utils.rnn.pad_sequence(x1, batch_first=True)
        x2 = torch.nn.utils.rnn.pad_sequence(x2, batch_first=True)
        x1 = x1.reshape(hparams.batch_size,max(len_x1),hparams.feats_dim) #prev 54
        x2 = x2.reshape(hparams.batch_size,max(len_x2),hparams.feats_dim)
        x1 = torch.nn.utils.rnn.pack_padded_sequence(x1,len_x1,batch_first=True, enforce_sorted=False)
        x2 = torch.nn.utils.rnn.pack_padded_sequence(x2,len_x2,batch_first=True, enforce_sorted=False)
        x1 = x1.cuda()
        x2 = x2.cuda()

        output, h1, h2, res = model(x1,x2,h1,h2)

        for i in range(hparams.batch_size):
            if (output[i]>hparams.sigmoid_threshold):
            #if (output[i]>hparams.euc_dist_threshold):
                lab=1
            else:
                lab=0
            test_predicted.append(lab)

    test_target = np.concatenate(test_target,axis=0)

    acc_calc(test_predicted,test_target)
    precision_recall(test_predicted,test_target)




    print('\nVALIDATION SET OF 10%')

    # Test phase after training
    model.eval()
    h = model.init_hidden(hparams.batch_size)
    h1 = h.data
    h2 = h.data

    test_target = []
    test_predicted = []

    for batch_idx, (x1,y1,x2,y2) in enumerate(validationloader):
        h1 = h.data
        h2 = h.data
        h1 = h1.cuda()
        h2 = h2.cuda()

        final_labels = np.abs(y1-y2)

        # invert labels (1- positive pair, 0 - negative pair)
        #final_labels = final_labels-1
        #final_labels = np.abs(final_labels)

        final_labels = final_labels.cuda()

        test_target.append(final_labels.cpu().numpy())

        #x1 = x1.reshape(hparams.batch_size,1,54)
        #x2 = x2.reshape(hparams.batch_size,1,54)

        x1 = return_batch_read(x1)
        x2 = return_batch_read(x2)
        
        for i in range(len(x1)):
            x1[i] = torch.Tensor(x1[i])
            x2[i] = torch.Tensor(x2[i])

        len_x1=[]
        len_x2=[]
        for i in range(len(x1)):
            len_x1.append(len(x1[i]))
            len_x2.append(len(x2[i]))


        x1 = torch.nn.utils.rnn.pad_sequence(x1, batch_first=True)
        x2 = torch.nn.utils.rnn.pad_sequence(x2, batch_first=True)
        x1 = x1.reshape(hparams.batch_size,max(len_x1),hparams.feats_dim) #prev 54
        x2 = x2.reshape(hparams.batch_size,max(len_x2),hparams.feats_dim)
        x1 = torch.nn.utils.rnn.pack_padded_sequence(x1,len_x1,batch_first=True, enforce_sorted=False)
        x2 = torch.nn.utils.rnn.pack_padded_sequence(x2,len_x2,batch_first=True, enforce_sorted=False)
        x1 = x1.cuda()
        x2 = x2.cuda()

        output, h1, h2, res = model(x1,x2,h1,h2)

        for i in range(hparams.batch_size):
            if (output[i]>hparams.sigmoid_threshold):
            #if (output[i]>hparams.euc_dist_threshold):
                lab=1
            else:
                lab=0
            test_predicted.append(lab)

    test_target = np.concatenate(test_target,axis=0)

    acc_calc(test_predicted,test_target)
    precision_recall(test_predicted,test_target)



    # plot train and validation losses
    plot_losses(train_loss_track,val_loss_track)
