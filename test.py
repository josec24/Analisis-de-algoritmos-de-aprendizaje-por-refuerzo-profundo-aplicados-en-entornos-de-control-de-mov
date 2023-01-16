import numpy as np
import gym

from agents.SAC_Agent import SAC_Agent

Envs = ['BipedalWalker-v3','BipedalWalkerHardcore-v3','HalfCheetah-v4','Humanoid-v4']
index=0

while(True):
    print('Ingrese el numero para indicar el entorno:\n1)BipedalWalker-v3\n2)BipedalWalkerHardcore-v3\n3)HalfCheetah-v4\n4)Humanoid-v4')
    index=int(input())
    if index==1 or index==2 or index==3 or index==4:
        break

index-=1      
nameEnv=Envs[index]
env = gym.make(nameEnv,render_mode='human')
ModelIdex=3500
model = SAC_Agent(env.observation_space.shape[0],env.action_space.shape[0])
#load trained model
model.load('SAC',ModelIdex,nameEnv)

max_action = float(env.action_space.high[0])

def Action_adapter(a,max_action):
    return  a*max_action

for j in range(5000000):
        state,_=env.reset()
        done, total_reward, steps =  False, 0, 0
        trunc=False
        while not ((done or trunc) or (steps >= 100000)):
            a = model.select_action(state, deterministic=True, with_logprob=False)
            act = Action_adapter(a, max_action)
            s_prime, reward, done,trunc, _ = env.step(act)

            if trunc:
                done=True

            total_reward += reward
            steps += 1
            state = s_prime
        