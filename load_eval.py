import os
import torch
import csv
import cv2
import numpy as np
import json
import pickle
import h5py

def writeText(img, text):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,10)
    fontScale              = 0.5
    fontColor              = (0,0,255)
    lineType               = 2

    img = cv2.putText(img,text,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)
    return img

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

def getlistname(h5array):
    t = np.array(h5array).reshape(-1)
    t = [i.decode('utf8') for i in t]
    t = [i[:i.find('.')].split('_') for i in t]
    t = ['{:0>5s}_{:0>2s}'.format(x[0],x[1]) for x in t]
    return np.array(t)

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
        self.evalAttn, self.full_attn = self.load_ATTN_result(data_dir, 'attn_scores.h5')
        self.evalHD, self.full_hd = self.load_HD_result(data_dir, 'hd_scores.json')

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

        ## data analize
        self.hd_images, self.index_hd, self.attn_images, self.index_attn = self.loadImage('Data/')

    def loadImage(self, dataDir):
        f_attn = h5py.File(dataDir + 'attn_images.h5', 'r')
        f_hd = h5py.File(dataDir + 'hd_images.h5', 'r')
        index_attn = np.arange(self.total*25)
        list_names = getlistname(f_attn['list_names'])

        index_attn = sorted(index_attn, key = lambda i:list_names[i])
        index_hd = np.arange(self.total *25)
        return f_hd['output_256'], index_hd, f_attn['gen_images'], index_attn

    def load_HD_result(self, dataDir, fileDir):
        a = json.load(open(dataDir + fileDir,'r'))
        arr = np.array(a['mean'])
        arr1 = arr.reshape((-1,25)).mean(1)
        arr1 = np.exp(arr1)
        return arr1, np.exp(arr)

    def load_ATTN_result(self, dataDir, fileDir):
        filename = dataDir + fileDir
        f = h5py.File(filename, 'r')
        arr  = np.array(f['scores'])
        arr = np.log(arr)
        arr1 = arr.reshape((-1,25)).mean(1)
        arr1 = np.exp(arr1)
        return arr1, np.exp(arr)

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
        res = sorted(perm, key = lambda x: self.full_attn[x] - self.full_hd[x])
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

    def showImage(self,Net , index):
        print(data.captions[index//25])
        embeddings = data.embedding[index//25]
        choose = Net(embedding)
        if (choose >= 0.5):
            choose = ''
        else :
            choose = ''
        
        img_attn = cv2.cvtColor(self.attn_images[self.index_attn[index]], 4)
        img_hd = cv2.cvtColor(self.hd_images[self.index_hd[index]], 4)
        if  choose == 'attn':
            img_hybrid = img_attn
        elif choose == 'hd':
            img_hybrid = img_hd
        
        img = np.concatenate([img_attn, img_hd, img_hybrid], 0)
        #img = writeText(img, data.captions[index//25]) 
        #cv2.imshow('hd'+ str(i), img)

        return img
#        while True :
#            t = cv2.waitKey(0)
#            print(t)
#            if (t == 110):
#                break
#            if (t == 115):
#                name = input()
#                fileDir = saveImgDir + name
#                os.makedirs(fileDir, exist_ok = True)
#                cv2.imwrite(fileDir + '/attn{}.png'.format(index), img_attn)
#                cv2.imwrite(fileDir + '/hd{}.png'.format(index), img_hd)
#                break
#                
#        cv2.destroyAllWindows()

    def getcaptionbyindex(captions, idx):
        return captions[idx//25]

if __name__ == '__main__':
    data = Data(64)
    perm = np.arange(data.full_hd.__len__())
    perm = data.sortbydiffent(perm)
    t = torch.load('Data/save/best/--description_flex_--hidden_1500_--lu_leaky_--final_activation_leaky_--batch_norm_True_--n_res_block_0_--n_fully_1_--learning_rate_1e-4_--init_xavier_True_--batch_size_64_--cost_func_MSE_best')
    t

    for i in range(0,0000,1):
        img1 = data.showImage(perm[i])
#        img2 = data.showImage(perm[i+1], i+1)
#        img3 = data.showImage(perm[i+2], i+2)
#        img = np.concatenate([img1,img2,img3], 0)

        saveImgDir = 'Data/savesingleIMG/'
        cv2.imwrite(saveImgDir + '/{}.png'.format(i), img1)
        
