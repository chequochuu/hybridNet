import csv
import cv2
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

def indexiformat(i, f_attn):
    x = f_attn['list_names'][i].decode('utf8').split('_')
    return '{:0>4s}_{}'.format(x[0],x[1])

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
        self.save_checkpoint_dir = data_dir + 'save/'
        self.log_dir = data_dir + 'log/'
        self.dataDir = data_dir
        self.evalAttn = self.load_ATTN_result(data_dir, 'attn_scores.h5')
        self.evalHD = self.load_HD_result(data_dir, 'hd_scores.json')

        self.total = self.evalHD.__len__()
        self.ntest = int(self.total*1/5)
        self.ntrain = self.total - self.ntest
        self.captions = load_captions(self.dataDir).reshape(-1) 
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

    def loadImage(self, dataDir):
        f_attn = h5py.File(dataDir + 'attn_images.h5', 'r')
        f_hd = h5py.File(dataDir + 'hd_images.h5', 'r')
        index_attn = np.arange(self.total*5)
        index_attn = sorted(index_attn, key = lambda i: indexiformat(i, f_attn))
        index_hd = np.arange(self.total *5)
        return f_hd['output_256'], index_hd, f_attn['gen_images'], index_attn

    def load_HD_result(self, dataDir, fileDir):
        a = json.load(open(dataDir + fileDir,'r'))
        arr = np.array(a['mean']).reshape((-1,25)).mean(1)
        arr = np.exp(arr)
        return arr

    def load_ATTN_result(self, dataDir, fileDir):
        filename = dataDir + fileDir
        f = h5py.File(filename, 'r')
        arr  = np.array(f['scores'])
        arr = np.log(arr)
        arr = arr.reshape((-1,25)).mean(1)
        arr = np.exp(arr)
        return arr

    def sortIndex(self):
        self.tevalAttn = sorted(self.tevalAttn, key = lambda entry: entry[0]) 
        self.tevalHD = sorted(self.tevalHD, key = lambda entry: entry[0]) 
        self.alreadySortedIndex = True

    def getcaption(self):
        cap_lists = []
        for i in range(self.ntest, self.pivotHDbetter):
            idx = self.permutation[i]
            cap_lists.append(self.captions[idx])
        return cap_lists

    def sortbydiffent(self, perm):
        res = sorted(perm, key = lambda x: self.evalAttn[x] - self.evalHD[x])
        return res

    def next(self):
        start = self.train_index1
        end = start + self.batch_size
        if end > self.total:
            np.random.shuffle(self.permutation[self.ntest: self.total])
            start = self.ntest
            end = start +  self.batch_size
        idx = self.permutation[start:end]
        self.train_index1 = end

        return self.embeddings[idx], self.evalHD[idx], self.evalAttn[idx]

    def next_test(self):
        start = self.test_index
        end = start + self.batch_size
        if end > self.ntest:
            start = 0
        end = start + self.batch_size
        self.test_index = end
        idx = self.permutation[start:end]
        return self.embeddings[idx], self.evalHD[idx], self.evalAttn[idx]

    def showImage(self,index):
        for i in range(25):
            t = index*25 + i
            img = cv2.cvtColor(self.Attn_images[t], 4)
            cv2.imshow('attn'+ str(i), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        for i in range(5):
            t = index*5 + i
            img = cv2.cvtColor(self.HD_images[t], 4)
            cv2.imshow('hd'+ str(i), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#    def getEmbedding(self, index)
    
if __name__ == '__main__':
    data = Data(64)
    zzz = data.getcaption()
    s1 = data.evalAttn
    s2 = data.evalHD

