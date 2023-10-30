import torch

def MultiLoss(X_embedding, X_output, label):

    task1_loss = weighted_mseloss(X_embedding, label)
    task2_loss = weighted_mseloss(X_output, label)

    return task1_loss, task2_loss


def weighted_mseloss(input,target):

    target = target.squeeze()

    a = torch.ones(input.shape).to("cuda")

    loss_vector = (input-target)**2

    m = int((a.shape[0]+1)/2)
    a[0:m] = 2

    return torch.dot(a,loss_vector)/a.shape[0]

