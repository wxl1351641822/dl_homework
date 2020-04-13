from datetime import datetime
import os
import pickle
import numpy as np
class config:
    def __init__(self):

        self.conv_struct=[[3],['M','relu',16],['G','relu',128]]
        self.conv_dropout = [0, 0, 0, 0]
        self.kernel_size=[[3],[3,5],[3]]
        self.filters = self.kernel_size[0]
        self.dim = int(32 / (2 ** (len(self.conv_struct)-1)))
        if (self.conv_struct[-1][0] == 'M' or self.conv_struct[-1][0] == 'A'):
            self.feature_dim = int(len(self.kernel_size[-1])*self.conv_struct[-1][-1] * self.dim * self.dim)
        else:
            self.feature_dim = len(self.kernel_size[-1])*self.conv_struct[-1][-1]

        self.fc = [self.feature_dim,1]
        self.fc_act=['tanh']*4
        self.fc_dropout=[]

        self.val_id=5

        self.gamma=0.9
        self.EPOCH=100
        self.LR=0.01
        self.BATCH_SIZE=16
        self.early_stopping=5*4
        self.time=str(datetime.now())

    def change_conv(self):
        # print(self.conv_struct)
        self.dim = int(32 / (2 ** (len(self.conv_struct)-1)))
        if (self.conv_struct[-1][0] == 'M' or self.conv_struct[-1][0] == 'A'):
            self.feature_dim = int(len(self.kernel_size[-1])*self.conv_struct[-1][-1] * self.dim * self.dim)
        else:
            self.feature_dim = len(self.kernel_size[-1])*self.conv_struct[-1][-1]
        self.fc[0]=self.feature_dim
        # print(self.dim,self.feature_dim)
        # print()
    #



    def string(self):
        s=self.get_title()+"| "+self.time+" | "+str(self.val_id)+" | "+str(0)+" | "+str(0)+" |"+str(0)+" | " +str(self.conv_struct)+" | "+str(self.kernel_size)+" | "+str([self.fc,self.fc_act])+" | "+str(self.conv_dropout+self.fc_dropout)+" | "+str(self.gamma)+" | "+str(self.EPOCH)+" | "+str(self.LR)+" | "+str(self.BATCH_SIZE)+" | "+str(self.early_stopping)+" |"
        return s
    def write_string(self,filepath,train_acc,max_acc,mean_acc):
        s="| "+self.time+" | "+str(self.val_id)+" | "+str(train_acc)+" | "+str(max_acc)+" | "+str(mean_acc)+" | "+str(self.conv_struct)+" | "+str(self.kernel_size)+" | "+str([self.fc,self.fc_act])+" | "+str(self.conv_dropout+self.fc_dropout)+" | "+str(self.gamma)+" | "+str(self.EPOCH)+" | "+str(self.LR)+" | "+str(self.BATCH_SIZE)+" | "+str(self.early_stopping)+" |\n"
        with open(filepath,'a',encoding='utf-8') as f:
            f.write(s)

    def get_title(self):
        s='| time | val_id | train_acc | max_acc | mean_acc | conv_struct | kernel_size | fc | dropout | gamma | EPOCH | LR | BATCH_SIZE | early_stopping |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n'
        return s
    def count_weight(self):
        data_dir = './cifar-10-batches-py'
        train_pathname = 'data_batch_'
        train_path = os.path.join(data_dir, train_pathname)
        for i in range(5):
            file=train_path + str(i + 1) + '_catdog'
            with open(file, 'rb') as f:
                t = pickle.load(f, encoding='bytes')
                # print(t)
                label = np.array(t['labels'])
                print(label[label==0].shape[0],label[label==1].shape[0])
# opt=config()
# opt.count_weight()