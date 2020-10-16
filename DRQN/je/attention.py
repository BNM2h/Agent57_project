import gym
import numpy as np
from network import Qnet2
import torch
import time
import cv2
import random
from memory import Memory
import torch.nn.functional as F
from collections import deque

memory=Memory(300,200)
env=gym.make('PongDeterministic-v4')
net=Qnet2().to('cuda')
target_net=Qnet2().to('cuda')
obs=env.reset()
epsilon=1.0
gamma=0.998
optimizer = torch.optim.Adam(net.parameters(),lr=0.005)
epsilon_min=0.001

def p_obs(obs):
    #210,160,3 to 3*160*210 -> 
    # img=np.zeros_like(obs)
    # for i in range(3):
    #     img[:,:,i]=obs[:,:,0]/2+obs[:,:,1]/6+obs[:,:,2]/3
    # cv2.imshow('obs',img)
    # cv2.waitKey(3)
    obs=obs.T
    res=np.zeros_like(obs)
    res=obs[0]/2+obs[1]/6+obs[2]/3
    #print(res.shape) 160*210
    return res/255.0

def get_action(obs,epsilon):
    temp=list(obs)
    states=torch.zeros((1,len(obs),160,210))
   # print(qvalue)
    if np.random.rand()>epsilon and len(obs)==4:
        for e,i in enumerate(temp):
            #states[0,len(obs)-e-1]=i
            states[0,e]=i
        qvalue=net(states.to('cuda')).cpu()
        _, action = torch.max(qvalue[0],0)
        action=action.item()
        #print(action)
    else:
        action=random.randint(0,5)
    return action
def epislon_decay(ep):
    return ep*0.992

def train():
    #print('train')
    batch_size=32
    sequence_length=4
    batch=memory.sample(batch_size,sequence_length)
    states = torch.stack(batch.state).view(batch_size*4, sequence_length,160,210)
    next_states = torch.stack(batch.next_state).view(batch_size*4, sequence_length,160,210)
    actions = torch.stack(batch.action).view(batch_size*4, sequence_length, -1).long()
    rewards = torch.stack(batch.reward).view(batch_size*4, sequence_length, -1)
    masks = torch.stack(batch.mask).view(batch_size*4, sequence_length, -1)
    pred= net(states.to('cuda')).to('cuda')
    pred = pred.gather(1,actions[:,-1,:].to('cuda'))
    next_pred=target_net(next_states.to('cuda'))
    next_pred=next_pred.max(-1, keepdim=True)[0].to('cuda')
    target = rewards[:,-1,:].to('cuda') + masks[:,-1,:].to('cuda') * gamma * next_pred
    loss = F.mse_loss(pred, target.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    del states, next_states,rewards,masks
        #torch.cuda.empty_cache()

def target_update():
    target_net.load_state_dict(net.state_dict())

for ep in range(1000):
    obs=env.reset()
    obs=p_obs(obs)
    obs=torch.from_numpy(obs).float().unsqueeze(0)
    done=False
    print('episode:',ep,'epsilon: ',epsilon)
    step=0
    score=0
    observations=deque(maxlen=4)
    while not done:
        step+=1
        env.render()
        observations.append(obs)
        action=get_action(observations,epsilon)
        next_obs,reward,done,info=env.step(action)
        next_obs=p_obs(next_obs)
        next_obs=torch.from_numpy(next_obs).float().unsqueeze(0)
        score+=reward
        if step>200:
            done=True
        mask= 1 if not done else 0
        memory.push(obs.squeeze(0), next_obs.squeeze(0), action, reward, mask)
        obs=next_obs
        if ep>40:
            train()
    if epsilon>epsilon_min and ep>20:
        epsilon=epislon_decay(epsilon)
    print('score: ', score)
    if (ep%20==0) and (ep!=0):
        target_update()