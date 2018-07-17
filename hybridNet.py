import sys
import argparse, os
from tqdm import tqdm
from load_eval import *
from NETutils import *

stopword = ['hybridNet.py']#, '--hidden_1', '--hidden_2', '--learning_rate'] 
allargvs = '' 
for i in sys.argv[:-4]:
    allargvs += (i+'_') if i not in stopword else ''

print(allargvs)

parser = argparse.ArgumentParser(description = 'Gans')    
parser.add_argument('--hidden', type=int, default=1000,
                    help='hidden size.')
parser.add_argument('--lu', type=str)
parser.add_argument('--final_activation', type=str)
parser.add_argument('--batch_norm', type=str2bool)
parser.add_argument('--n_res_block', type=int, default=0)
parser.add_argument('--n_fully', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--init_xavier', type=str2bool)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--cost_func', type=str)
parser.add_argument('--reuse_weight',type=str2bool)
parser.add_argument('--iter_load', type=int, default=10000)

args = parser.parse_args()

input_dim = 1024
output_dim = 1
total_steps = 1000000
print_step = 50
save_step = 1000
batch_size = args.batch_size
lr = args.learning_rate 
hidden_dim = args.hidden
if args.cost_func=='BCE':
    cost_func = nn.BCELoss() #MSELoss()
else:
    cost_func = nn.MSELoss()
    
if (torch.cuda.is_available()):
    device = 'cuda'
else:
    device = 'cpu'
N = hybridNN(batch_size, 1024,hidden_dim, 1, args)
data = Data(batch_size)
N = N.to(device) 
optim = torch.optim.Adam(N.parameters(), lr = lr) 

begin_step = 0 
best_score = 0
if args.reuse_weight == True:
    begin_step = args.iter_load+1
    checkpoint = load_checkpoint(data.dataDir, allargvs+str(args.iter_load))
    N = checkpoint['N']
    optim = checkpoint['optim']
    best_score = checkpoint['best_score']

for i in tqdm(range(begin_step, total_steps)):
    embeddings, inceptionsHD, inceptionsAttn = data.next()
    embeddings = makeEmbedding(embeddings, device)
    tam = inceptionsAttn > inceptionsHD
    results = Variable(torch.Tensor([[1] if i else [0] for i in tam])).to(device)
    outs = N(embeddings) 
    loss = cost_func(outs, results)
    loss.backward()
    optim.step()
    N.zero_grad()
    if (i % print_step == 0):
        hyb = [inceptionsHD[i] if outs[i]<0.5 else inceptionsAttn[i] for i in range(outs.__len__())]
        ground_truth = [inceptionsHD[i] if results[i]<0.5 else inceptionsAttn[i] for i in range(outs.__len__())]
        hyb = np.array(hyb)
        ground_truth = np.array(ground_truth)
        print('loss: {}, HD: {}, ATTN: {}, hybrid_train:{}, ground_truth: {}'.format(loss, inceptionsHD.sum(), inceptionsAttn.sum(), hyb.sum(), ground_truth.sum()))

        embeddings, inceptionsHD, inceptionsAttn = data.next_test()
        embeddings = makeEmbedding(embeddings, device)
        tam = inceptionsAttn > inceptionsHD
        results = Variable(torch.Tensor([[1] if i else [0] for i in tam])).to(device)
        outs = N(embeddings)
        hyb = [inceptionsHD[i] if outs[i]<0.5 else inceptionsAttn[i] for i in range(outs.__len__())]
        ground_truth = [inceptionsHD[i] if results[i]<0.5 else inceptionsAttn[i] for i in range(outs.__len__())]
        hyb = np.array(hyb)
        ground_truth = np.array(ground_truth)
        print('loss: {}, HD: {}, ATTN: {}, hybrid:{}, ground_truth: {}'.format(loss, inceptionsHD.sum(), inceptionsAttn.sum(), hyb.sum(), ground_truth.sum()))
        score = fullTest(N,data, batch_size, device) 
        if (i % print_step  == 0):
            save_checkpoint(N, optim,args, score, data.dataDir, allargvs+str(i)) 
        
        if (score > best_score):
            print(best_score)
            best_score = score
            save_checkpoint(N, optim,args, score, data.dataDir+'best/', allargvs+'best') 
            
