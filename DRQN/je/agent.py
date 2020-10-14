import gym
import numpy as np
from network import Qnet
import torch
import time
import cv2
import random
from memory import Memory
import torch.nn.functional as F

memory=Memory(1000,300)
env=gym.make('Pong-v0')
net=Qnet().to('cuda')
target_net=Qnet().to('cuda')
obs=env.reset()
epsilon=1.0
gamma=0.98
optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
epsilon_min=0.001

def p_obs(obs):
    #210,160,3 to 3*160*210 -> 
    # img=np.zeros_like(obs)
    # for i in range(3):
    #     img[:,:,i]=obs[:,:,2-i]
    # cv2.imshow('obs',img)
    # cv2.waitKey(3)
    obs=obs.T
    return obs/255.0

def get_action(obs,epsilon):
    qvalue=net(obs).cpu()
   # print(qvalue)
    if np.random.rand()>epsilon:
        _, action = torch.max(qvalue[0],1)
        action=action.item()
        #print(action)
    else:
        action=random.randint(0,5)
    return action
def epislon_decay(ep):
    return ep*0.995

def train():
    try:
    #print('train')
        batch_size=16*2
        sequence_length=8
        batch=memory.sample(batch_size,sequence_length)
        states = torch.stack(batch.state).view(batch_size, sequence_length,3,160,210)
        next_states = torch.stack(batch.next_state).view(batch_size, sequence_length,3,160,210)
        actions = torch.stack(batch.action).view(batch_size, sequence_length, -1).long()
        rewards = torch.stack(batch.reward).view(batch_size, sequence_length, -1)
        masks = torch.stack(batch.mask).view(batch_size, sequence_length, -1)
        pred= net(states,train=True).to('cuda')
        pred = pred.gather(2, actions.to('cuda'))
        next_pred=target_net(next_states,train=True)
        next_pred=next_pred.max(-1, keepdim=True)[0].to('cuda')
        target = rewards.to('cuda') + masks.to('cuda') * gamma * next_pred
        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #torch.cuda.empty_cache()
    except RuntimeError:
        pass

def target_update():
    target_net.load_state_dict(net.state_dict())

for ep in range(3000):
    obs=env.reset()
    obs=p_obs(obs)
    obs=torch.from_numpy(obs).float().unsqueeze(0)
    done=False
    print('episode:',ep,'epsilon: ',epsilon)
    step=0
    score=0
    while not done:
        step+=1
        env.render()
        action=get_action(obs,epsilon)
        next_obs,reward,done,info=env.step(action)
        next_obs=p_obs(next_obs)
        next_obs=torch.from_numpy(next_obs).float().unsqueeze(0)
        score+=reward
        if step>300:
            done=True
        mask= 1 if not done else 0
        memory.push(obs.squeeze(0), next_obs.squeeze(0), action, reward, mask)
        obs=next_obs
        if ep>20:
            train()
    net.reset()
    if epsilon>epsilon_min:
        epsilon=epislon_decay(epsilon)
    print('score: ', score)
    if (ep%20==0) and (ep!=0):
        target_update()
