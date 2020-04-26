import torch
import torch.nn.functional as F
import torch.nn as nn
class hand_number(nn.Module):
    def __init__(self,opt):
        super(hand_number,self).__init__()
        self.convs=nn.ModuleList([nn.Sequential(
            nn.Conv2d(
                in_channels=opt.conv_struct[i-1],#灰度图
                out_channels=opt.conv_struct[i],
                kernel_size=opt.kernel_size[i],
                stride=opt.stride[i],#步长
                padding=(opt.kernel_size[i]-opt.stride[i])//2#padding=(kernel_size-stride)/2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=opt.pool[i]),#2*2的窗口
            nn.Dropout(opt.conv_dropout[i])
        ) for i in range(1,len(opt.conv_struct))])#输出为[batch,16,14,14]
        dim=28.
        for i in range(1,len(opt.pool)):
            dim=int(dim/float(opt.pool[i]))
        feature_dim=opt.conv_struct[-1]*int(dim)*int(dim)

        self.prediction=nn.Linear(feature_dim,10)

    def forward(self,x):
        for conv in self.convs:
            x=conv(x)
        # x=self.conv(x)
        x=x.view(x.size(0),-1)
        # print(x.shape)
        output=self.prediction(x)
        return output
