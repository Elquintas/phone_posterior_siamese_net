import numpy as np

lr = 0.0001    #0.001
#init_lr = 0.01
epochs = 5  #24
batch_size = 32   #128

n_gru_layers = 2 #2 CHANGED CHANGED CHANGED
hidden_dim = 100 #100 because of bidirectional
dropout = 0.0 #0.5   # 0.2

bidirectional=True

rnn_dropout = 0.0 #0.25

feats_dim=35     #36

checkpoint_interval = 1
euc_dist_threshold = 0.5 #0.5
sigmoid_threshold = 0.5 ####################0.50


