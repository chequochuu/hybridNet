import csv
import numpy as np
import json
import pickle
import h5py
def str2bool(x):
    if x == 'True':
        return True
    elif x == 'False':
        return False
    else:
        raise(ValueError('must be True or False'))

def formatname(arr):
    for (i,j) in enumerate(arr):
        x = j[0].split('_')
        arr[i][0] = '{:0>4s}_{}'.format(x[0],x[1])
    return arr

def makePair(i,j,k):
    return [str(i)+'_'+str(j),k]

def load_eval_result(filename, datatype):
    if (datatype == 'json'):
        t = json.load(open(filename,'r'))['mean']
        ret = [makePair(i,j,t[i*5+j]) for i in range(2933) for j in range(5)]
    elif datatype == 'csv':
        with open(filename,'r') as f:
            ret = []
            t = csv.reader(f)
            for i in t:
                ret.append(i)
        ret=ret[1:]
    else:
        raise ValueError('wtf')

    ret = formatname(ret)
    return ret

def load_captions(data_dir, filename = 'caption_array_extend.pickle'):
    a = pickle.load(open(data_dir + filename,'rb'))
    return a

def load_embeddings(data_dir, filename = 'original.h5'):
    f = h5py.File(data_dir+ filename, 'r')
    ret = list(f['embedding'])
    ret = np.array(ret)
    return ret

# the first ntest use for test, the rest use for train
class Data():
    def __init__(self, batch_size, data_dir = 'Data/'):
        self.dataDir = data_dir
        self.tevalAttn = load_eval_result(data_dir + 'eval_2933_5_tien.csv', 'csv')
        self.tevalHD = load_eval_result(data_dir + 'birds_256_G_epoch_500_inception_score.json', 'json')
        self.sortIndex()
        self.id = np.array([i[0] for i in self.tevalHD])
        self.evalHD = np.array([i[1] for i in self.tevalHD])
        self.evalHD = self.evalHD.astype(np.float32)
        self.evalAttn = np.array([i[1] for i in self.tevalAttn])
        self.evalAttn = self.evalAttn.astype(np.float32)
        self.total = self.evalHD.__len__()
        self.ntest = int(self.total*1/5)
        self.ntrain = self.total - self.ntest
        self.captions = load_captions(self.dataDir)  
        self.embeddings = load_embeddings(self.dataDir)
        self.alreadySortedIndex = False
        self.batch_size = batch_size
        self.train_index1 = self.ntest
        self.test_index = 0
        np.random.seed(1234)
        self.permutation = np.random.permutation(self.total)
        self.permutation[self.ntest:] = self.sortbydiffent(self.permutation[self.ntest:])
        self.pivotHDbetter = self.ntest
        while self.evalHD[self.permutation[self.pivotHDbetter]] > self.evalAttn[self.permutation[self.pivotHDbetter]]:
            self.pivotHDbetter += 1
        self.train_index2 = self.pivotHDbetter
        

    def sortIndex(self):
        self.tevalAttn = sorted(self.tevalAttn, key = lambda entry: entry[0]) 
        self.tevalHD = sorted(self.tevalHD, key = lambda entry: entry[0]) 
        self.alreadySortedIndex = True
#
#    def sortValue_1col(col):
#        if col == 'HD':
#            sorted(self.evalHD, key = lambda entry: entry[1]) 
#        elif col == 'ATTN' :
#            sorted(self.evalAttn, key = lambda entry: entry[1]) 


#    def getSentence(index, captions):
#        t = index.split('_')
#        return captions[int(t[0])][int(t[1])] 

    def sortbydiffent(self, perm):
        res = sorted(perm, key = lambda x: self.evalAttn[x] - self.evalHD[x])
        return res

    def next(self):
        start = self.train_index1
        end = start +(self.batch_size//2)
        if end > self.pivotHDbetter:
            start = self.ntest
            end = start + self.batch_size//2
#            shuffle data
#            perm = np.random.permutation(self.pivotHDbetter - s)
#            self.permutation[self.ntest:] = self.permutation[perm + self.ntest]
        idx = self.permutation[start:end]
        self.train_index1 = end

        start = self.train_index2
        end = start + self.batch_size//2
        if end > self.total:
            start = self.pivotHDbetter
            end = start + self.batch_size//2
        idx2 = self.permutation[start:end]
        self.train_index2 = end

        concatted_idx = np.concatenate([idx,idx2])
        return self.embeddings[concatted_idx], self.evalHD[concatted_idx], self.evalAttn[concatted_idx]

    def next_test(self):
        start = self.test_index
        end = start + self.batch_size
        if end > self.ntest:
            start = 0
        end = start + self.batch_size
        self.test_index = end
        idx = self.permutation[start:end]
        return self.embeddings[idx], self.evalHD[idx], self.evalAttn[idx]

#    def getEmbedding(self, index)
    
if __name__ == '__main__':
    data = Data(64)
    s1 = data.evalAttn
    s2 = data.evalHD

