import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal
import numpy as np
from torch.autograd import Variable
FLOAT = torch.FloatTensor

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet,self).__init__()
        #input 210*160*3 output 4 action
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4) 
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        #self.fc1 = nn.Linear(11264, 512)
        self.lstm = nn.LSTM(11264, 512)
        self.fc2 = nn.Linear(512, 6)
        self.hx=Variable(torch.zeros(1,1,512)).type(FLOAT).to('cuda')
        self.cx=Variable(torch.zeros(1,1,512)).type(FLOAT).to('cuda')

        #nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.kaiming_normal_(self.fc2.weight.data)
        # #nn.init.kaiming_normal_(self.fc3.weight.data)
        # nn.init.kaiming_normal_(self.fc5.weight.data)

    def forward(self, x,train=False):
        x=x.to('cuda')
        batch_size=x.size(0)
        sequence_length=x.size(1)
        if not train:
            x=x.view((-1,3,160,210))
            x=F.relu(self.bn1(self.conv1(x)))
            x=F.relu(self.bn2(self.conv2(x)))
            x=F.relu(self.bn3(self.conv3(x))) #64,7,10
           # x=x.view(batch_size,-1)
           # x=F.relu(self.fc1(x))
           # x=x.unsqueeze(1)
            x=x.view((1,1,11264))
            x,(self.hx,self.cx) =self.lstm(x,(self.hx,self.cx))
            x=self.fc2(x)
            return x
        else:
            hx=Variable(torch.zeros(1,1,512)).type(FLOAT).to('cuda')
            cx=Variable(torch.zeros(1,1,512)).type(FLOAT).to('cuda')
            sequence_length=x.size(1)
            x=x.view((batch_size*sequence_length,3,160,210))
            x=F.relu(self.bn1(self.conv1(x)))
            x=F.relu(self.bn2(self.conv2(x)))
            x=F.relu(self.bn3(self.conv3(x))) #64,7,10
            #x=x.view(batch_size*sequence_length,-1)
            #x=F.relu(self.fc1(x))
            #x=x.unsqueeze(1)
            feature=x.view((batch_size,sequence_length,11264))
            qvalue=torch.zeros((batch_size,sequence_length,6))    
            for i in range(sequence_length):
                x=feature[:,i,:]
                x=x.unsqueeze(1)
                x,(hx,cx)=self.lstm(x,(hx,cx))  
                x = self.fc2(x)
                qvalue[:,i,:]=x.squeeze(1)  
            return qvalue    
        
    
    def reset(self,done=True):
        if done==True:
           self.cx=Variable(torch.zeros(1,1,512)).type(FLOAT).to('cuda')
           self.hx=Variable(torch.zeros(1,1,512)).type(FLOAT).to('cuda')
        else:
            self.cx=Variable(self.cx.data).type(FLOAT).to('cuda')
            self.hx=Variable(self.hx.data).type(FLOAT).to('cuda')


class Qnet2(nn.Module):
    def __init__(self):
        super(Qnet2,self).__init__()
        #input 210*160*3 output 4 action
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4) 
        # self.bn1 = nn.BatchNorm2d(32)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.lstm = nn.LSTM(11264, 512)
        # self.fc2 = nn.Linear(512, 6)
        self.c1 = nn.Conv2d(4, 32, 8, stride=4)
        self.attention_layer = MultiHeadAttention(32)
        self.c2 = nn.Conv2d(32, 64, 4, stride=2)
        self.c3 = nn.Conv2d(64, 32, 3, stride=1)
        self.l1 = nn.Linear(11264, 512)
        self.l2 = nn.Linear(512, 6)

    def forward(self, x):
        batch_size=x.size(0)
        sequence_length=x.size(1)
        h = F.relu(self.c1(x))
        h = self.attention_layer(h, h, h)
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        h = h.reshape(-1).view(batch_size, 11264)
        h = F.relu(self.l1(h))
        h = self.l2(h)
        return h
    

class MultiHeadAttention(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.w_qs = nn.Conv2d(size, size, 1)
        self.w_ks = nn.Conv2d(size, size, 1)
        self.w_vs = nn.Conv2d(size, size, 1)

        self.attention = ScaledDotProductAttention()

    def forward(self, q, k, v):
        residual = q
        #print(residual.size())
        q = self.w_qs(q).permute(0, 2, 3, 1)
        k = self.w_ks(k).permute(0, 2, 3, 1)
        v = self.w_vs(v).permute(0, 2, 3, 1)

        attention = self.attention(q, k, v).permute(0, 3, 1, 2)
        #print(q.size(),attention.size())
        out = attention + residual
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        attn = torch.matmul(q, k.transpose(2, 3))
        output = torch.matmul(attn, v)

        return output