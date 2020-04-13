from torchvision import datasets
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from cnn import hand_number as MODEL
import  torchvision.transforms as transforms
from config import config
from tqdm import tqdm
from datetime import datetime
import numpy as np
# BATCH_SIZE=64
# LR=0.0002
# EPOCH=5
now=0
init_seed = 1222
log_path='./log/log'

torch.manual_seed(init_seed)
# torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed) # 用于numpy的随机数
train_set = datasets.MNIST('./MNIST_data', train=True, transform=transforms.ToTensor(), download=True)

train_set = torch.utils.data.random_split(train_set, [10000]*6)
print(len(train_set))
import matplotlib.pyplot as plt
# print(train_set.train_data.size())
# print(train_set.train_labels.size())
def getImg(data,label):
    plt.imshow(data.numpy(),cmap='gray')
    plt.title('%i' %label)
    plt.show()
# getImg(train_set.train_data[0],train_set.train_labels[0])
def metric(y_true,pred):
    # print(y_true)
    # print(pred)
    TP=torch.tensor([0.]*10)
    FP=torch.tensor([0.]*10)
    FN=torch.tensor([0.]*10)
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
def one(opt,val_id):
    opt.time = str(now) + '_' + str(datetime.now())
    print(
        "---------------------------------------------------------------------------------------------------------------------------")
    print(opt.string())
    print(
        "---------------------------------------------------------------------------------------------------------------------------")
    # BATCH_NUM = 50000 // opt.BATCH_SIZE
    # test_x = torch.unsqueeze(test_set.test_data, dim=1)
    # print(test_x.shape)
    # test_y = test_set.test_labels
    model=MODEL(opt)
    #大数据常用Adam优化器，参数需要model的参数，以及学习率
    optimizer = torch.optim.Adam(model.parameters(), opt.LR)
    #定义损失函数，交叉熵
    loss_func = nn.CrossEntropyLoss()
    max_val_acc=0.0
    early_stopping=opt.early_stopping
    max_train_acc=0.0

    for epoch in range(opt.EPOCH):
        for i in range(6):
            if(i==val_id):
                continue
            train_loader = Data.DataLoader(dataset=train_set[i], batch_size=opt.BATCH_SIZE, shuffle=True, num_workers=0)

            total_loss = 0.0
            total_mean_F1=0.0
            train_accuracy=0.0
            for step,(batch_x,batch_y) in tqdm(enumerate(train_loader)):
                epoch_best=0.0
                output=model(batch_x)
                loss=loss_func(output,batch_y)
                total_loss += loss.data
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred_y = torch.argmax(output, -1)
                # print(pred_y.shape)
                # print(step,float((pred_y == batch_y).sum()) / float(batch_y.size(0)))
                train_accuracy = float((pred_y == batch_y).sum()) / float(batch_y.size(0))
                F1, mean_F1 = metric(batch_y.long(), pred_y.long())
                if (train_accuracy > max_train_acc):
                    max_train_acc = train_accuracy
                    total_mean_F1 = mean_F1
                    # print(output.size())

        with torch.no_grad():
            accuracy = 0.0
            l = 0.0
            val_loss = 0.0
            val_loader = Data.DataLoader(dataset=train_set[val_id], batch_size=10000, shuffle=False, num_workers=0)
            for x, y in val_loader:
                val_output = model(x)
                p_y = torch.argmax(val_output, -1)
                # print(pred_y.shape)
                accuracy = accuracy + float((p_y == y).sum())
                l += float(y.size(0))
                t_loss = loss_func(val_output, y)
                val_loss = val_loss + t_loss.data
                F1, mean_F1 = metric(y.long(), p_y.long())
            accuracy /= l


            # print("epoch:", epoch, "| step:", step, "|val accuracy：%.4f" % accuracy)

            if(max_val_acc<accuracy):
                max_val_acc=accuracy
            else:
                early_stopping-=1
            print("| epoch:", epoch, "| train loss:%.4f" % total_loss,
                  "|train accuracy：%.4f" % max_train_acc, "|train mean_F1：%.4f" % total_mean_F1,
                  "|max val accuracy：%.4f" % max_val_acc, "| val loss:%.4f" % val_loss,
                  "|val accuracy：%.4f" % accuracy, "|val F1：", F1.data, "|val mean_F1：%.4f" % mean_F1.data,
                  " |")
        if(early_stopping<0):
            break
    return max_val_acc,mean_F1.data,max_train_acc
def predict(model):
    test_set = datasets.MNIST('./MNIST_data', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=10000, shuffle=True, num_workers=0)
    with torch.no_grad():
        accuracy = 0.0
        l = 0.0
        for x, y in test_loader:
            test_output = model(x)
            p_y = torch.argmax(test_output, -1)
            # print(pred_y.shape)
            accuracy = accuracy + float((p_y == y).sum())
            l += float(y.size(0))
        accuracy /= l
        print( "test accuracy：%.4f" % accuracy)
def cross_validate(opt):
    val_acc=0.0
    F1=0.0
    acc=0.0
    for i in range(6):
        max_acc,mean_F1,train_acc=one(opt,i)
        val_acc+=max_acc
        F1+=mean_F1
        acc+=train_acc

    return val_acc/6.,F1/6.,acc/6.
def main():
    opt=config()
    global now
    ####动一下conv-channel
    # cross_validate(opt)
    s="""###########################################################################################\n
    实验2：channel数目的影响\n
    ###########################################################################################\n"""
    opt.print_experiment(log_path,s)
    layer=3

    for last_layer in [16,32,64]:
        for layer in [2,3,4]:
            l=(last_layer-1)//(layer-1)
            opt.conv_struct=[]
            for i in range(layer-1):
                opt.conv_struct.append(1+i*l)
            opt.conv_struct.append(last_layer)
            opt.kernel_size=[3]*layer
            opt.pool=[2]*layer
            opt.stride=[1]*layer
            opt.conv_dropout=[0]*layer
            # print(last_layer,layer,opt.conv_struct)
            max_acc,mean_F1,train_acc=cross_validate(opt)
            opt.write_string(log_path, train_acc, max_acc,mean_F1)
            now+=1
    s = """###########################################################################################\n
        实验1：channel大小的影响\n
        ###########################################################################################\n"""
    opt.print_experiment(log_path, s)
    layer = 3
    for last_layer in [16, 32, 64]:
        opt.conv_struct = [1, (1 + last_layer) // 2, last_layer]
        opt.kernel_size = [3] * layer
        opt.pool = [2] * layer
        opt.stride = [1] * layer
        opt.conv_dropout = [0] * layer
        max_acc, mean_F1, train_acc = cross_validate(opt)
        opt.write_string(log_path, train_acc, max_acc, mean_F1)
        now+=1
# print(output.size())
if __name__=='__main__':
    main()