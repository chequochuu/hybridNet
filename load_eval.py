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
        self.log_dir = data_dir +'log/'
        self.dataDir = data_dir
        self.tevalAttn = load_eval_result(data_dir + 'eval_2933_5_tien1.csv', 'csv')
        self.tevalHD = load_eval_result(data_dir + 'birds_256_G_epoch_500_inception_score_25_1.json', 'json') 
        self.sortIndex() 
        self.id = np.array([i[0] for i in self.tevalHD])
        self.evalHD = np.array([i[1] for i in self.tevalHD])
        self.evalHD = self.evalHD.astype(np.float32)
        self.evalAttn = np.array([i[1] for i in self.tevalAttn])
        self.evalAttn = self.evalAttn.astype(np.float32)
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
        self.HD_images, self.HD_images_index, self.Attn_images, self.Attn_images_index = self.loadImage()


    def loadImage(self):
        f_attn = h5py.File('attn_images.h5', 'r')
        f_hd = h5py.File('hd_images.h5', 'r')
        index_attn = np.arange(self.total*5)
        index_attn = sorted(index_attn, key = lambda i: indexiformat(i, f_attn))
        index_hd = np.arange(self.total *5)
        return f_hd['output_256'], index_hd, f_attn['gen_images'], index_attn

        

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
        HDBetterPortion = self.batch_size*1//2
        AttnBetterPortion = self.batch_size - HDBetterPortion
        start = self.train_index1
        end = start + HDBetterPortion
        if end > self.pivotHDbetter:
#            shuffle data
#            perm = np.random.permutation(self.pivotHDbetter - s)
#            self.permutation[self.ntest:] = self.permutation[perm + self.ntest]
            np.random.shuffle(self.permutation[self.ntest: self.pivotHDbetter])
            start = self.ntest
            end = start +  HDBetterPortion
        idx = self.permutation[start:end]
        self.train_index1 = end

        start = self.train_index2
        end = start + AttnBetterPortion 
        if end > self.total:
            np.random.shuffle(self.permutation[self.pivotHDbetter:])
            start = self.pivotHDbetter
            end = start + AttnBetterPortion
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

    def showImage(self,index):
        for i in range(5):
            t = index*5 + i 
            cv2.imshow('attn'+ str(i), self.Attn_images[self.Attn_images_index[t]])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        for i in range(5):
            t = index*5 + i 
            cv2.imshow('hd'+ str(i), self.HD_images[self.HD_images_index[t]])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#    def getEmbedding(self, index)
    
if __name__ == '__main__':
    data = Data(64)
    zzz = data.getcaption()
    s1 = data.evalAttn
    s2 = data.evalHD
    perm = np.arange(data.total)
    perm = data.sortbydiffent(perm)
    for i in range(100):
        print('HD: {}, ATTN: {}'.format(data.evalHD[perm[i]], data.evalAttn[perm[i]]))
        print(data.captions[perm[i]])
        data.showImage(perm[i])

