import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
from tqdm import tqdm
from config import config
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
# BATCH_SIZE=64
# LR=0.01
# EPOCH=100
init_seed = 1222
torch.manual_seed(init_seed)
# torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed) # 用于numpy的随机数

data_dir='./cifar-10-batches-py'
meta_path=os.path.join(data_dir,'batches.meta')
test_path=os.path.join(data_dir,'test_batch')
train_pathname='data_batch_'
log_path='./log/log'
train_path = os.path.join(data_dir, train_pathname )
now=0

def metric(y_true,pred):
    # print(y_true)
    # print(pred)
    TP=torch.tensor([0.]*2)
    FP=torch.tensor([0.]*2)
    FN=torch.tensor([0.]*2)
    for y,p in zip(y_true,pred):
        if(y==p):
            TP[y]+=1
        else:
            FP[p]+=1
            FN[y]+=1
    # print(TP,FP,FN)
    # print(TP+FP)
    precision=TP/(TP+FP+1e-4)
    recall=TP/(TP+FN+1e-4)
    # print(precision,recall)
    F1=2*precision*recall/(precision+recall+1e-4)
    mean_F1=torch.mean(F1)
    return F1,mean_F1



def unpickle(file):
    with open(file,'rb') as f:
        dict=pickle.load(f,encoding='bytes')
    return dict

def pickle_save(file,t):
    with open(file,'wb') as f:
        dict=pickle.dump(t,f)
    return dict

# print(labellist)
def get_img(data,label):
    img = np.reshape(data, (3, 32, 32))
    img = img.transpose((1, 2, 0))
    plt.imshow(img)
    if(label==0):
        plt.title('cat' )
    else:
        plt.title('dog')
    plt.show()
def get_catdog():
    train=[]
    cat_id=3
    dog_id=5
    for i in range(5):
        # train_path=os.path.join(data_dir,train_pathname+str(i+1))
        t=unpickle(train_path+str(i + 1)+'_catdog')
        # print(t[b'labels'])
        t[b'labels']=np.array(t[b'labels'])
        index=np.where((t[b'labels']==cat_id) | (t[b'labels']==dog_id))
        # print(index)
        # t[b'labels'][(t[b'labels']==cat_id) | (t[b'labels']==dog_id)])
        t[b'labels']=t[b'labels'][index]
        t[b'labels'][t[b'labels'] == cat_id] = 0
        t[b'labels'][t[b'labels'] == dog_id] = 1
        t[b'labels']= t[b'labels'].tolist()
        t[b'data']=t[b'data'][index]
        # print(t[b'labels'].shape, t[b'data'].shape)
        # t[b'labels']=[1 if x==cat_id else 2 if x==dog_id else 0 for x in t[b'labels']]
        # print(t[b'labels'])
        # print(np.where(t[b'labels']>3 ))
        new_t = {}
        new_t['labels'] = t[b'labels']
        new_t['data'] = t[b'data']
        pickle_save(train_path+'_catdog',new_t)
        train.append(t)
    meta=unpickle(meta_path)
    test=unpickle(test_path)
    # print(t)
    # print(test)
    # labellist=meta[b'label_names']
    # get_img(test['data'].reshape(-1,3,32,32)[3],test['labels'][3])
    test[b'labels']=np.array(test[b'labels'])
    index=np.where((test[b'labels']==cat_id) | (test[b'labels']==dog_id))
    test[b'labels']=test[b'labels'][index]
    test[b'labels'][test[b'labels']==cat_id]=0
    test[b'labels'][test[b'labels']==dog_id]=1
    test[b'labels']= test[b'labels'].tolist()
    test[b'data']=test[b'data'][index]

    new_test={}
    new_test['labels']=test[b'labels']
    new_test['data']=test[b'data']
    pickle_save(test_path+'_catdog',new_test)
    print(len(test[b'labels']))
# print(test)


# print(test[b'labels'].shape,test[b'data'].shape)
# print(train[0][b'data'].shape)
# print(type(train[0][b'labels']))
# get_img(test[0]['data'].reshape(-1,3,32,32)[3],test[0]['labels'][3])
# print(Data.TensorDataset(torch.from_numpy(i[b'data'].reshape(10000,3,32,32)),torch.tensor(i[b'labels'])))
# train_loaders=[Data.DataLoader(dataset= Data.TensorDataset(torch.from_numpy(i[b'data'].reshape(-1,3,32,32)),torch.FloatTensor(i[b'labels'])),batch_size=BATCH_SIZE,shuffle=True,num_workers=0) for i in train[:-1]]
# val_loader=Data.DataLoader(dataset= Data.TensorDataset(torch.from_numpy(train[-1][b'data'].reshape(-1,3,32,32)),torch.FloatTensor(train[-1][b'labels'])),batch_size=train[-1][b'data'].shape[0],shuffle=True,num_workers=0)
# test_loader=Data.DataLoader(dataset=Data.TensorDataset(torch.from_numpy(test[b'data'].reshape(-1,3,32,32)),torch.FloatTensor(test[b'labels'])),batch_size=test[b'data'].shape[0],shuffle=True,num_workers=0)
def predict(model):
    with torch.no_grad():
        accuracy = 0.0
        l = 0.0
        test = unpickle(test_path + '_catdog')
        test_loader = Data.DataLoader(
            dataset=Data.TensorDataset(torch.from_numpy(test['data'].reshape(-1, 3, 32, 32)),
                                       torch.FloatTensor(test['labels'])), batch_size=test['data'].shape[0], shuffle=False,
            num_workers=0)
        for x, y in test_loader:
            test_output = model(x)
            # p_y = torch.argmax(test_output, -1)
            test_output = test_output.squeeze(-1)
            test_output[test_output > 0] = 1
            test_output[test_output <= 0] = 0
            p_y=test_output
            # print(pred_y.shape)
            accuracy = accuracy + float((p_y == y).sum())
            l += float(y.size(0))
        accuracy /= l
        print("test accuracy：%.4f" % accuracy)


# def get_label_data(train,label):
#     data={"data":[],"label":[]}
#     for t in train:
#         for img,l in zip(t[b'data'],t[b'labels']):
#             if(labellist[l]==label):
#
# # get_img(train[-1][b'data'][0],train[-1][b'labels'][0])
#
# get_label_data(train, b'cat')
class cifar10_classify(nn.Module):
    def __init__(self,config):
        super(cifar10_classify,self).__init__()

        def pool(x):
            if (x == 'M'):
                return nn.MaxPool2d(kernel_size=2)
            elif (x == 'A'):
                return nn.AvgPool2d(kernel_size=2)
            else:
                return nn.MaxPool2d(kernel_size=config.dim*2)

        def activate(x):
            if x == 'relu':
                return nn.ReLU()
            elif x == 'tanh':
                return nn.Tanh()

        self.convs2 = \
            nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=config.conv_struct[i - 1][-1] * len(config.kernel_size[i - 1]),
                            out_channels=config.conv_struct[i][-1],
                            kernel_size=k,
                            stride=1,
                            padding=(k - 1) // 2  # padding=(kernel_size-stride)/2
                        ),
                        activate(config.conv_struct[i][1]),
                        nn.BatchNorm2d(config.conv_struct[i][-1]),
                        pool(config.conv_struct[i][0])
                    )  # 2*2的窗口
                    for k in config.kernel_size[i]
                ])
                for i in range(1,len(config.conv_struct))
            ] )#输出为[batch,32，8,8]
        self.dropout=nn.ModuleList([nn.Dropout(i) for i in config.conv_dropout])
        # print(config.fc)
        self.fc=nn.ModuleList(
            [
                    nn.Sequential(
                        nn.Linear(config.fc[i-1],config.fc[i]),
                        activate(config.fc_act[i]),
                        nn.Dropout(config.fc_dropout[i]))
                    if i+2<len(config.fc)
                    else
                    nn.Linear(config.fc[i - 1], config.fc[i])
                for i in range(1,len(config.fc))
            ]
        )



    def forward(self,x):
        # print(x.shape)
        # print(x)

        i=0
        x=x.float()/255.
        for i,modulse in enumerate(self.convs2):
            x=[conv(x) for conv in modulse]
            # print(x.shape)
            # xs=[]
            # for conv in modulse:
            #     # print(conv)
            #     xs +=[conv(x) ]
                # print(xs[-1].shape)

            x=torch.cat(x, 1)
            x = self.dropout[i](x)


        x = x.view(x.size(0), -1)
        # print(x.shape)
        for f in self.fc:
            x=f(x)
        # x=self.dropout1(x)
        # x=self.bn1(x)
        # print(x.shape)
        # x = self.conv1(x.float()/255.)
        # x=self.conv2(x)
        # x = self.dropout2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.dropout3(x)

        # print(x.shape)
        # x=self.fc1(x)
        # output=self.fc2(x)
        # output=self.prediction(x)
        return x

def one_fold(opt):
    opt.time=str(now)+'_'+str(datetime.now())
    print("---------------------------------------------------------------------------------------------------------------------------")
    print(opt.string())
    print("---------------------------------------------------------------------------------------------------------------------------")
    BATCH_SIZE=opt.BATCH_SIZE
    model=cifar10_classify(opt)
    #大数据常用Adam优化器，参数需要model的参数，以及学习率
    optimizer = torch.optim.Adam(model.parameters(), opt.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=opt.gamma)
    #定义损失函数，交叉熵
    loss_func = nn.BCEWithLogitsLoss()
    val_id=opt.val_id
    max_val_acc=0.
    max_train_acc=0.
    early_stopping=opt.early_stopping
    # print(opt.EPOCH)
    for epoch in range(opt.EPOCH):
        scheduler.step()
        i=0
        for i in range(1,3):
            if(i+1==val_id):
                continue
            train = unpickle(train_path+str(i + 1)+'_catdog')
            train_loader=Data.DataLoader(dataset= Data.TensorDataset(torch.from_numpy(train['data'].reshape(-1,3,32,32)),torch.FloatTensor(train['labels'])),batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
            i+=1
            total_loss=0.0
            train_accuracy=0.0
            step=0
            total_mean_F1=0.0
            for step,(batch_x,batch_y) in enumerate(train_loader):

                output = model(batch_x)
                # print(output)

                output=output.squeeze(-1)
                # print(output.shape)
                # print(batch_y.shape)
                loss = loss_func(output, batch_y)
                total_loss+=loss.data
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(output)
                output[output>0]=1
                output[output <= 0] = 0
                pred_y=output
                # print(pred_y.shape)
                # print(pred_y)
                train_accuracy=train_accuracy+ float((pred_y == batch_y).sum()) / float(batch_y.size(0))
                F1, mean_F1 = metric(batch_y.long(), pred_y.long())
                # print(F1,mean_F1)
                total_mean_F1= total_mean_F1+mean_F1
            with torch.no_grad():
                accuracy = 0.0
                l = 0.0
                val_loss=0.0
                # val_path = os.path.join(data_dir, train_pathname + str(5) + '_catdog')
                val = unpickle(train_path+str(val_id)+'_catdog')
                val_loader = Data.DataLoader(
                    dataset=Data.TensorDataset(torch.from_numpy(val['data'].reshape(-1, 3, 32, 32)),
                                               torch.FloatTensor(val['labels'])), batch_size=val['data'].shape[0], shuffle=False,
                    num_workers=0)

                for x, y in val_loader:
                    val_output = model(x)
                    val_output = val_output.squeeze(-1)
                    val_output[val_output > 0] = 1
                    val_output[val_output <= 0] = 0
                    p_y = val_output

                    # print(pred_y.shape)
                    t_loss = loss_func(val_output, y)
                    val_loss=val_loss+t_loss.data
                    accuracy = accuracy + float((p_y == y).sum())
                    F1,mean_F1=metric(y.long(), p_y.long())
                    l += float(y.size(0))
                accuracy /= l
                # print("epoch:", epoch, "| step:", step, "|val accuracy：%.4f" % accuracy)
                train_accuracy/=step
                total_mean_F1/=step
                if(train_accuracy>max_train_acc):
                    max_train_acc=train_accuracy
                if(accuracy>max_val_acc):
                    max_val_acc=accuracy
                    early_stopping = opt.early_stopping
                else:
                    early_stopping-=1
                print("| epoch:", epoch,  "| filename:",i,"| train loss:%.4f" % total_loss,
                      "|train accuracy：%.4f" % train_accuracy,"|train mean_F1：%.4f" % total_mean_F1.data, "|max val accuracy：%.4f" %  max_val_acc,"| val loss:%.4f" % val_loss, "|val accuracy：%.4f" % accuracy,"|val F1：" , F1.data,"|val mean_F1：%.4f" % mean_F1.data," |")
        if(early_stopping<=0):
            print("early_stopping")
            break
    return max_val_acc,max_train_acc
def cross_validate(opt):
    mean_acc=0.
    for i in range(2,3):
        opt.val_id=i+1
        max_acc,train_acc=one_fold(opt)
        mean_acc=mean_acc+max_acc
        opt.write_string(log_path,train_acc,max_acc,mean_acc/(i+1.))
    return mean_acc


def fc_grid(opt1):
    global now
    for i in range(1, 3):  # fc
        ll = []
        # print(opt1.string())
        opt1.change_conv()
        l = (opt1.fc[0] - opt1.fc[-1]) // i
        # print(l)
        # print(opt1.fc[0],opt1.fc[1])
        for j in range(i):
            ll.append(opt1.fc[0] - j * l)
            # print(ll)
        ll.append(opt1.fc[-1])
        opt1.fc = ll
        if(now<13):#4
            now += 1
            continue
        cross_validate(opt1)
        now += 1


def main():
    opt=config()
    # with open(log_path, 'w', encoding='utf-8') as f:
    #     f.write(opt.get_title())
    kernel = [3, 5, 7]
    pool=['M','A','G']

    # cross_validate(opt)
    print('###########################################################################################')
    print('实验1：前馈层的层数+多尺寸+卷积层数+不同类型的池化')
    print('###########################################################################################')

    # opt1.fc=[opt.fc[0],opt.fc[-1]]
    # [feature,1]
    # cross_validate(opt1)
    # [feature,64,1]
    global now
    now=0

    for k in range(2, 4):#卷积深度
        opt1 = config()
        opt1.kernel_size = opt.kernel_size[:k]
        opt1.conv_struct = opt.conv_struct[:k]
        for z in range(1, 3):  # 第一层卷积的宽度
            opt1.kernel_size[1] = kernel[:z]  #
            # print(opt1.string())
            if (len(opt1.conv_struct) == 3):  # pool
                for a in range(2):
                    opt1.conv_struct[1][0] = pool[a]
                    for b in range(3):
                        opt1.conv_struct[2][0] = pool[b]
                        fc_grid(opt1)
            elif (len(opt1.conv_struct) == 4):
                for a in range(2):
                    opt1.conv_struct[1][0] = pool[a]
                    for b in range(2):
                        opt1.conv_struct[2][0] = pool[b]
                        for c in range(3):
                            opt1.conv_struct[3][0] = pool[c]
                            fc_grid(opt1)
            else:
                for a in range(3):
                    opt1.conv_struct[1][0] = pool[a]
                    fc_grid(opt1)


    print('###########################################################################################')
    print('实验2：前馈层的层数+多尺寸+卷积层数')
    print('###########################################################################################')
    opt1 = config()
    # opt1.fc=[opt.fc[0],opt.fc[-1]]
    # [feature,1]
    # cross_validate(opt1)
    # [feature,64,1]
    for i in range(1, 3):
        ll = []
        l = (opt.fc[-1] - opt.fc[0]) // i
        for j in range(i):
            ll.append(opt.fc[0] + i * l)
        ll.append(opt.fc[-1])
        opt1.fc = ll
        for k in range(1, 3):
            opt1.kernel_size = opt.kernel_size[:k]
            opt1.conv_struct = opt.conv_struct[:k]
            for z in range(1,3):
                opt1.kernel_size[0] = kernel[:z]
                cross_validate(opt1)

    print('###########################################################################################')
    print('实验3：前馈层的层数+多尺寸+卷积层数')
    print('###########################################################################################')
    opt1 = config()
    # opt1.fc=[opt.fc[0],opt.fc[-1]]
    # [feature,1]
    # cross_validate(opt1)
    # [feature,64,1]
    for i in range(1, 3):
        ll = []
        l = (opt.fc[-1] - opt.fc[0]) // i
        for j in range(i):
            ll.append(opt.fc[0] + j * l)
        ll.append(opt.fc[-1])
        opt1.fc = ll
        for k in range(1, 3):
            opt1.kernel_size[0] = kernel[:k]
            cross_validate(opt1)
    print('###########################################################################################')
    print('实验4：前馈层的层数+卷积层数')
    print('###########################################################################################')
    opt1 = config()
    # opt1.fc=[opt.fc[0],opt.fc[-1]]
    # [feature,1]
    # cross_validate(opt1)
    # [feature,64,1]
    for i in range(1, 3):
        ll = []
        l = (opt.fc[-1] - opt.fc[0]) // i
        for j in range(i):
            ll.append(opt.fc[0] + j * l)
        ll.append(opt.fc[-1])
        opt1.fc = ll
        for k in range(1, 3):
            opt1.kernel_size = opt.kernel_size[:k]
            opt1.conv_struct = opt.conv_struct[:k]
            cross_validate(opt1)
        # cross_validate(opt1)
    print('###########################################################################################')
    print('实验5：前馈层的层数')
    print('###########################################################################################')
    opt1=config()
    # opt1.fc=[opt.fc[0],opt.fc[-1]]
    # [feature,1]
    # cross_validate(opt1)
    #[feature,64,1]
    for i in range(1,3):
        ll=[]
        l=(opt.fc[-1]-opt.fc[0])//i
        for j in range(i):
            ll.append(opt.fc[0]+i*l)
        ll.append(opt.fc[-1])
        opt1.fc=ll
        cross_validate(opt1)
    print('###########################################################################################')
    print('实验6：卷积核层+深')
    print('###########################################################################################')
    # kernel = [3, 5, 7]
    for i in range(1, 3):
        for j in range(1,3):
            opt1 = config()
            opt1.kernel_size = opt.kernel_size[:i]
            opt1.conv_struct = opt.conv_struct[:i]
            opt1.kernel_size[0] = kernel[:i]
            cross_validate(opt1)
    ##第一个实验：卷积层数的影响
    print('###########################################################################################')
    print('实验7：卷积层数的影响')
    print('###########################################################################################')
    for i in range(1,3):
        opt1 = config()
        opt1.kernel_size=opt.kernel_size[:i]
        opt1.conv_struct=opt.conv_struct[:i]
        cross_validate(opt1)
    print('###########################################################################################')
    print('实验8：多尺寸卷积核的影响')
    print('###########################################################################################')
    for i in range(1,3):
        opt1 = config()
        opt1.kernel_size[0]=kernel[:i]
        cross_validate(opt1)
    # print('###########################################################################################')
    # print('实验9：不同激活层的影响')
    # print('###########################################################################################')
    # for i in range(1, 3):
    #     opt1 = config()
    #     opt1.kernel_size[0] = kernel[:i]
    #     cross_validate(opt1)







if __name__=='__main__':

    main()


