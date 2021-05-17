from torch import nn
import torch
import torch.nn.functional as F

class MoCo(nn.Module):
    def __init__(self, base_network, K, T, m, device):
        '''
        :param base_network: Base classifier
        :param K: Queue size
        :param T: temperature
        :param m: momentum
        :param device: 'cpu' or 'cuda'
        '''
        super(MoCo, self).__init__()
        self.K = K
        self.T = T
        self.m = m
        self.device = device
        self.f_k = base_network()
        self.f_q = base_network()

        # MoCoV2 improvement - change fc layer to 2-layer MLP
        dim0, dim1 = self.f_q.fc.weight.shape
        self.f_q = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Linear(dim1, dim0))
        dim0, dim1 = self.f_k.fc.weight.shape
        self.f_k = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Linear(dim1, dim0))

        # Init to same weights and turn of gradient calculation for f_k
        for p_k, p_q in zip(self.f_k.parameters(), self.f_q.parameters()):
            p_k.data.copy_(p_q)
            p_k.requires_grad = False

    def forward(self, x_q, x_k, queue, return_k=False):
        q = F.normalize(self.f_q(x_q), dim=1)

        with torch.no_grad():
            shuffle_index, unshuffle_index = shuffle_batch(x_q.shape[0], self.device)
            k = F.normalize(self.f_k(x_k[shuffle_index]))
            k = k[unshuffle_index]

        N, C = q.shape

        l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1))
        l_neg = torch.mm(q.view(N, C), queue.clone().detach().view(C, self.K))

        logits = torch.cat([l_pos, l_neg], dim=1) / self.T

        labels = torch.zeros(N).to(self.device)

        for p_k, p_q in zip(self.f_k.parameters(), self.f_q.parameters()):
            p_k.data = self.m * p_k + (1.0-self.m) * p_q

        if return_k:
            return logits, labels, k

        return logits, labels

def shuffle_batch(batch_size, device):
    shuffle_index = torch.randperm(batch_size).long().to(device)
    unshuffle_index = torch.zeros(batch_size).long().to(device)
    arange = torch.arange(batch_size).long().to(device)
    unshuffle_index.index_copy_(0, shuffle_index, arange)
    return shuffle_index, unshuffle_index

