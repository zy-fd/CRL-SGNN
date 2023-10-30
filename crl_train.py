import numpy as np
import torch.optim as optim
import scipy.io as sci
import torch
import random
import time
from tqdm import *
import scipy.sparse as ssp
from CRL_SGNN import CRL_SGNN
from loss import MultiLoss
from crl_test import mean_error
# The demo is for networks of 1000 nodes. For other network sizes, the dimensionality of the fully connected layer needs to be modified to match the length of the controllability robustness sequence
def val(model,test_A,test_x,test_label,device):
    model.eval()
    loss_val1 = 0
    loss_val2 = 0
    output_final=[]
    val_mean_error = 0
    for i in range(len(test_A[0])):
       a = test_A[:,i][0]
       x = test_x[i,:,:]
       x = x.astype(float)
       x = x / 999
       x = torch.FloatTensor(x).to(device)

       # edge_index
       B = ssp.spmatrix.tocoo(a)
       edge_index = [B.row, B.col]
       edge_index = np.array(edge_index)
       edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

       task1_output,task2_output = model(x,edge_index)

       m = torch.FloatTensor(test_label[i,:]).to(device)
       loss_test_task1,loss_test_task2= MultiLoss(task1_output,task2_output,m)

       loss_val1 += loss_test_task2.cpu().item()
       loss_val2 += loss_test_task1.cpu().item()
       val_mean_error += mean_error(task2_output,m)/len(test_A[0])
       output_final.append(task2_output.cpu().detach().numpy())
    model.train()
    return loss_val1, output_final,loss_val2,val_mean_error


def train(epochs,device):
    # load data
    train_data_x = sci.loadmat('data/train_data/train/train_degree.mat')['feature_gather']
    train_A = sci.loadmat('data/train_data/train/train_A.mat')['adj']
    train_data_label = sci.loadmat('data/train_data/train/train_label.mat')['label']

    val_A = sci.loadmat('data/train_data/val/val_A.mat')['adj']
    val_x = sci.loadmat('data/train_data/val/val_degree.mat')['feature_gather']
    val_label = sci.loadmat('data/train_data/val/val_label.mat')['label']

    # load model
    model = CRL_SGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-5)
    num_train_ins = len(train_data_x)
    best_test_loss = float('inf')
    shuffle_list = random.sample(range(0, num_train_ins), num_train_ins)  # shuffle

    for epoch in range(epochs):

        loss_train_sum = 0
        begin = time.time()
        for i in trange(len(train_A[0])):

                # degree_feature
                x = train_data_x[shuffle_list[i],:,:]
                x = x.astype(float)
                # depends on networks size:N-1
                x = x/999
                x = torch.FloatTensor(x)

                A = train_A[0,shuffle_list[i]]
                # edge_index
                B = ssp.spmatrix.tocoo(A)
                edge_index = [B.row, B.col]
                edge_index = np.array(edge_index)
                edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

                # CRL
                task1_output,task2_output = model(x.to(device),edge_index)
                label = train_data_label[shuffle_list[i],:]
                label = torch.FloatTensor(label)

                # loss
                task1_loss,task2_loss = MultiLoss(task1_output,task2_output,label.to(device))

                loss_train = task1_loss + 2 * task2_loss
                loss_train_sum += task2_loss.cpu().item()

                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

        stop = time.time()
        print("epoch_spend_time:{:.3f}".format(stop-begin))

        with torch.no_grad():
          val_loss, output_final,val_loss1,val_mean_error = val(model,val_A,val_x,val_label,device)

        if (val_loss < best_test_loss):
            print("best_loss_val: {:.4f}".format(val_loss))
            best_test_loss = val_loss
            torch.save(model.state_dict(), 'model_pth/crl_sgnn_ra.pth')
            print("mean_error: {:.4f}".format(val_mean_error))
        print("Epoch: {}".format(epoch + 1),
                      "loss_train: {:.4f}".format(loss_train_sum))

        print("val_loss: {:.4f},val_loss_branch: {:.4f}".format(val_loss,val_loss1))

if __name__ == "__main__":

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   train(20,device)
