from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from load_eval import *

#load = False
load = True
batch_size = 64
lr = 1e-3
input_dim = 1024
output_dim = 1
total_steps = 10000
save_step = 1000
hidden_dim1 = 1800
hidden_dim2 = 1150
cost_func = nn.BCELoss() #MSELoss()

class hybridNN(torch.nn.Module):
    def __init__(self, batch_size, input_dim, output_dim):
        super(hybridNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1) 
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2) 
        self.fc3 = nn.Linear(hidden_dim2, output_dim) 
    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

def save_checkpoint(N, optim, data_dir, filename='checkpoint'):
    state = {'Net': N.state_dict(),
            'optim': optim.state_dict()}
    torch.save(state, data_dir + filename)
    
def load_checkpoint(data_dir, filename='checkpoint'):
    checkpoint = torch.load(data_dir + filename)
    return checkpoint['Net'], checkpoint['optim']

def makeEmbedding(embeddings):
    return Variable(torch.from_numpy(embeddings).float()).to(device)

N = hybridNN(batch_size, 1024, 1)
data = Data(batch_size)

if (torch.cuda.is_available()):
    device = 'cuda'
else:
    device = 'cpu'
N = N.to(device) 
optim = torch.optim.Adam(N.parameters(), lr = lr) 
if load == True:
    stateN, stateOp = load_checkpoint(data.dataDir)
    N.load_state_dict(stateN)
    optim.load_state_dict(stateOp)

for i in tqdm(range(total_steps)):
    embeddings, inceptionsHD, inceptionsAttn = data.next()
    embeddings = Variable(torch.from_numpy(embeddings).float()).to(device)
    tam = inceptionsAttn > inceptionsHD
    results = Variable(torch.Tensor([[1] if i else [0] for i in tam])).to(device)
    outs = N(embeddings) 
    loss = cost_func(outs, results)
    loss.backward()
    optim.step()
    N.zero_grad()
    if (i % save_step ==0):
        embeddings, inceptionsHD, inceptionsAttn = data.next_test()
        embeddings = makeEmbedding(embeddings)
        tam = inceptionsAttn > inceptionsHD
        results = Variable(torch.Tensor([[1] if i else [0] for i in tam])).to(device)
        hyb = [inceptionsHD[i] if outs[i]<0.5 else inceptionsAttn[i] for i in range(outs.__len__())]
        ground_truth = [inceptionsHD[i] if results[i]<0.5 else inceptionsAttn[i] for i in range(outs.__len__())]
        hyb = np.array(hyb)
        ground_truth = np.array(ground_truth)
        print('loss: {}, HD: {}, ATTN: {}, hybrid:{}, ground_truth: {}'.format(loss, inceptionsHD.sum(), inceptionsAttn.sum(), hyb.sum(), ground_truth.sum()))
        save_checkpoint(N, optim, data.dataDir) 

