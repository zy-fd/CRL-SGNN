import time
import networkx as nx
import scipy.io as sci
import numpy as np
import scipy.sparse as ssp
import torch
from CRL_SGNN import CRL_SGNN

def mean_error(output,label):

    error = torch.abs(output-label)
    meanerror = error.sum()/len(output)

    return meanerror

def load_data(m,n,test_data):

    test_A = []
    feature_gather = []
    label = []
    for i in range(100):
        degree = []
        A = test_data[m,n,i][0,0][0]
        label.append(test_data[m,n,i][0,0][1])
        test_A.append(A)
        G = nx.DiGraph(A)
        # in-degree
        in_degree = np.array(G.in_degree)[:,1]/999
        degree.append(in_degree)
        # out-degree
        out_degree = np.array(G.out_degree)[:,1]/999
        degree.append(out_degree)

        degree = np.array(degree)
        feature_gather.append(degree.transpose())
    feature_gather = np.array(feature_gather)
    label = np.array(label).squeeze()

    return feature_gather,label,test_A


def test(model,test_A,test_x,test_label,device):

    mean_error1 = 0
    output_final=[]

    for i in range(len(test_A)):

       a = test_A[i]

       x = torch.FloatTensor(test_x[i]).to(device)

       B = ssp.spmatrix.tocoo(a)
       edge_index = [B.row, B.col]
       edge_index = np.array(edge_index)
       edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

       model.eval()
       task1_output,task2_output = model(x,edge_index)

       model.train()

       m = torch.FloatTensor(test_label[i,:]).to(device)

       mean_error1+= mean_error(task2_output,m)

       output_final.append(task2_output.cpu().detach().numpy())

    output_final = np.array(output_final)

    return output_final,mean_error1


if __name__ == "__main__":

     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

     test_data = sci.loadmat('data/test_data/data_test.mat')['data_test']

     model = CRL_SGNN().to(device)

     model.load_state_dict(torch.load('model_pth/crl_sgnn_ra.pth'))

     num = 4

     mean_error_final = np.zeros((4,num))

     output_final = []

     loss_sum = 0
     loss_sum1 = 0

     for i in range(4):

        for j in range(num):

            m = i
            n = j
            begin = time.time()
            test_x,test_label,test_A = load_data(m,n,test_data)

            output,mean_error1 = test(model,test_A,test_x,test_label,device)
            end = time.time()

            output_final.append(output)

            mean_error_final[i,j] = mean_error1/100

            print("%.5f" % (mean_error1/100))