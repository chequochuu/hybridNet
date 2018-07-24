import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
def init_weight(m):
    classname =  m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_activate_func(func_name):
    if (func_name == 'leaky'):
        func = F.leaky_relu
    elif func_name == 'elu':
        func = F.elu
    elif func_name == 'relu':
        func = F.relu
    elif func_name == 'sigmoid':
        func = F.sigmoid
    else:
        ValueError('wrong function name {}'.format(func_name))
    return func

def downsample(x, stride):
    shape = x.shape
    return x.reshape(shape[0],shape[1]*stride, shape[2]/stride)

# 2 conv layer
class res_block(nn.Module):
    def __init__(self, in_channels, in_dims):
        super(res_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels*2, 3, 2, 1)
    #    self.pooling0 = nn.MaxPool1d(3,2,1)
        self.bn1 = nn.BatchNorm1d(in_channels*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels*2, in_channels*4,3, 2, 1)
  #      self.pooling1 = nn.MaxPool1d(3,2,1)
        self.bn2 = nn.BatchNorm1d(in_channels*4)
        self.out_channels = in_channels*4
        self.out_dims = in_dims/4

    def forward(self, x):
        residual = x
        out = self.conv1(x)
  #      out = self.pooling0(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
  #      out = self.pooling1(out)
        out = self.bn2(out)
        residual = downsample(residual, 4)
        out += residual
        out = self.relu(out)
        return out

class fully_block(nn.Module):
    def __init__(self, in_dim, hidden_dim, index ): #
        super(fully_block, self).__init__()
        if (index == 0): #from inp to hidden
            self.fc = nn.Linear(in_dim, hidden_dim)
        else:
            self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x)



class hybridNN(torch.nn.Module):
    def __init__(self, batch_size, input_dim,hidden_dim , output_dim, args):
        super(hybridNN, self).__init__()

        self.dropout = nn.Dropout(0.2)
        self.batch_size = batch_size
        #each block include 2 conv layer
        if (args.n_res_block>0):
            self.res = [res_block(2**(i*2), 2**(10-(i*2))) for i in range(args.n_res_block)]
            self.res = nn.Sequential(*self.res)
            res_out_dim = self.res[-1].out_dims
            res_out_channels = self.res[-1].out_channels
        else:
            res_out_dim = 1024
            res_out_channels = 1

        if (args.n_fully>0):
            self.fc = [fully_block(res_out_dim,hidden_dim, i) for i in range(args.n_fully)]
            self.fc = nn.Sequential(*self.fc)
            out_dim = hidden_dim
        else:
            out_dim = res_out_dim

        self.lu = get_activate_func(args.lu)

        self.bn = nn.BatchNorm1d(res_out_channels)

        self.final_fc = nn.Linear(out_dim, output_dim)
        self.final_fc1 = nn.Linear(res_out_channels, output_dim)
        self.final_activation = get_activate_func(args.final_activation)
        self.args = args
        if (args.init_xavier):
            self.apply(init_weight)

    def forward(self, x):
        out = self.dropout(x)
        for i in range(self.args.n_res_block):
            out = self.res[i](out)
        for i in range(self.args.n_fully):
            out = self.fc[i](out)
            out = self.lu(out)
        if (self.args.batch_norm):
            out = self.bn(out)
        out = self.final_fc(out)
        out = self.final_activation(out)
        out = out.reshape(self.batch_size, -1)
        out = self.final_fc1(out)
        out = self.final_activation(out)
        return out

def save_checkpoint(N, optim,args, score, data_dir, filename):
    state = {'Net': N,
            'optim': optim,
            'args': args,
            'score': score}
    torch.save(state, data_dir + filename)
    
def load_checkpoint(data_dir, filename='checkpoint'):
    checkpoint = torch.load(data_dir + filename)
    return checkpoint

def makeEmbedding(embeddings, device):
    return Variable(torch.from_numpy(embeddings.reshape((-1,1,1024))).float()).to(device)

def fullTestTrain(N, data, batch_size, device):
    total_HD = 0
    total_Attn = 0
    total_hyb = 0
    total_ground_truth = 0
    for i in range(int(data.ntest/batch_size)):
        embeddings, inceptionsHD, inceptionsAttn = data.next()
        embeddings = makeEmbedding(embeddings, device)
        tam = inceptionsAttn > inceptionsHD
        results = Variable(torch.Tensor([[1] if i else [0] for i in tam])).to(device)
        outs = N(embeddings) 
        hyb = [inceptionsHD[i] if outs[i]<0.5 else inceptionsAttn[i] for i in range(outs.__len__())]
        ground_truth = [inceptionsHD[i] if results[i]<0.5 else inceptionsAttn[i] for i in range(outs.__len__())]
        hyb = np.array(hyb)
        ground_truth = np.array(ground_truth)
        total_HD += inceptionsHD.sum()
        total_Attn += inceptionsAttn.sum()
        total_hyb += hyb.sum()
        total_ground_truth += ground_truth.sum()
        #print('loss: {}, HD: {}, ATTN: {}, hybrid_train:{}, ground_truth: {}'.format(loss, inceptionsHD.sum(), inceptionsAttn.sum(), hyb.sum(), ground_truth.sum()))

    print('HD: {}, ATTN: {}, hybrid:{}, ground_truth: {}'.format(total_HD, total_Attn, total_hyb, total_ground_truth))

def fullTest(N, data, batch_size, device):
    total_HD = 0
    total_Attn = 0
    total_hyb = 0
    total_ground_truth = 0
    for i in range(int(data.ntest/batch_size)):
        embeddings, inceptionsHD, inceptionsAttn = data.next_test()
        embeddings = makeEmbedding(embeddings, device)
        tam = inceptionsAttn > inceptionsHD
        results = Variable(torch.Tensor([[1] if i else [0] for i in tam])).to(device)
        outs = N(embeddings) 
        hyb = [inceptionsHD[i] if outs[i]<0.5 else inceptionsAttn[i] for i in range(outs.__len__())]
        ground_truth = [inceptionsHD[i] if results[i]<0.5 else inceptionsAttn[i] for i in range(outs.__len__())]
        hyb = np.array(hyb)
        ground_truth = np.array(ground_truth)
        total_HD += inceptionsHD.sum()
        total_Attn += inceptionsAttn.sum()
        total_hyb += hyb.sum()
        total_ground_truth += ground_truth.sum()
        #print('loss: {}, HD: {}, ATTN: {}, hybrid:{}, ground_truth: {}'.format(loss, inceptionsHD.sum(), inceptionsAttn.sum(), hyb.sum(), ground_truth.sum()))

    print('HD: {}, ATTN: {}, hybrid:{}, ground_truth: {}'.format(total_HD, total_Attn, total_hyb, total_ground_truth))
    return total_hyb
