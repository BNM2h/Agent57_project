import gym
import numpy as np
from network import Qnet,target_Qnet
import torch
import time
import cv2
import random
from memory import Memory
import torch.nn.functional as F

memory=Memory(100,300)
env=gym.make('Pong-v0')
net=Qnet().to('cuda:0')
target_net=target_Qnet()
obs=env.reset()
gamma=0.99
optimizer = torch.optim.Adam(net.parameters(),lr=0.001)

def p_obs(obs):
    #210,160,3 to 3*160*210 -> 
    # img=np.zeros_like(obs)
    # for i in range(3):
    #     img[:,:,i]=obs[:,:,2-i]
    # cv2.imshow('obs',img)
    # cv2.waitKey(3)
    obs=obs.T
    return obs

def get_action(obs,epsilon):
    qvalue=net(obs.to('cuda:0')).cpu()
    if np.random.rand()>epsilon:
        _, action = torch.max(qvalue[0],1)
        action=action.item()
        #print(action)
    else:
        action=random.randint(0,5)
    return action
def epislon_decay(ep):
    return ep*0.98

def train():
    batch_size=16
    sequence_length=5
    batch=memory.sample(batch_size,sequence_length)
    states = torch.stack(batch.state).view(batch_size, sequence_length,3,160,210)
    next_states = torch.stack(batch.next_state).view(batch_size, sequence_length,3,160,210)
    actions = torch.stack(batch.action).view(batch_size, sequence_length, -1).long()
    rewards = torch.stack(batch.reward).view(batch_size, sequence_length, -1)
    masks = torch.stack(batch.mask).view(batch_size, sequence_length, -1)
    pred= net(states.to('cuda:0'),train=True)
    pred = pred.gather(1, actions.to('cuda:0'))
    next_pred=target_net(next_states)
    next_pred=next_pred.max(-1, keepdim=True)[0]#.to('cuda:0')
    target = rewards + masks * gamma * next_pred
    loss = F.mse_loss(pred, target.to('cuda').detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()

def target_update():
    target_net.load_state_dict(net.state_dict())
def main():
    epsilon=1.0
    for ep in range(3000):
        obs=env.reset()
        obs=p_obs(obs)
        obs=torch.from_numpy(obs).float().unsqueeze(0)
        done=False
        print('episode:',ep,'epsilon: ',epsilon)
        step=0
        while not done:
            step+=1
            env.render()
            action=get_action(obs,epsilon)
            next_obs,reward,done,info=env.step(action)
            next_obs=p_obs(next_obs)
            next_obs=torch.from_numpy(next_obs).float().unsqueeze(0)
            if step>300:
                done=True
            mask= 1 if not done else 0
            memory.push(obs.squeeze(0), next_obs.squeeze(0), action, reward, mask)
            obs=next_obs
            if ep>20:
                train()
        net.reset()
        if (ep%10==0) and (ep!=0):
            target_update()
            epsilon=epislon_decay(epsilon)
            
if __name__ == '__main__':
    main()