import numpy as np
import gym

from agents.SAC_Agent import SAC_Agent
from data.helper import guardarPuntuacion
from buffer.batch import RandomBuffer

def Action_adapter(a,max_action):
    return  a*max_action

def Action_adapter_reverse(act,max_action):
    return  act/max_action

def Reward_adapter(reward, index):
    if index == 0 or index == 1:
        if reward <= -100: reward = -1
    return reward

def evaluate_policy(env, model, steps_per_epoch, max_action):
    scores = 0
    turns = 3
    for j in range(turns):
        state,_=env.reset()
        done, total_reward, steps =  False, 0, 0
        trunc=False
        while not ((done or trunc) or (steps >= steps_per_epoch)):
            a = model.select_action(state, deterministic=True, with_logprob=False)
            act = Action_adapter(a, max_action) 
            s_prime, reward, done,trunc, _ = env.step(act)

            if trunc:
                done=True

            total_reward += reward
            steps += 1
            state = s_prime
        scores += total_reward
    return scores/turns

if __name__ == '__main__':
    Envs = ['BipedalWalker-v3','BipedalWalkerHardcore-v3','HalfCheetah-v4','Humanoid-v4']
    index=0
    
    while(True):
        print('Ingrese el numero para indicar el entorno:\n1)BipedalWalker-v3\n2)BipedalWalkerHardcore-v3\n3)HalfCheetah-v4\n4)Humanoid-v4')
        index=int(input())
        if index==1 or index==2 or index==3 or index==4:
            break

    index-=1      
    nameEnv=Envs[index]
    Env_With_Dead = [True, True, True, True]
    Loadmodel=False
    ModelIdex=2000
    #env = gym.make(nameEnv)
    env = gym.make(nameEnv,render_mode='human')
    eval_env = gym.make(nameEnv)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_steps = env._max_episode_steps
    T_horizon = 2048
    Max_train_steps = 50000000
    save_interval = 5e5
    eval_interval = 5e3

    start_steps = 5*max_steps 
    update_after = 2*max_steps 
    update_every= 50


    model = SAC_Agent(env.observation_space.shape[0],env.action_space.shape[0])
    if Loadmodel: model.load('SAC',ModelIdex,nameEnv)

    replay_buffer = RandomBuffer(state_dim, action_dim, Env_With_Dead, max_size=int(1e6))


    total_steps = 0
    record=0
    rewards = []
    game=0
    while total_steps < Max_train_steps:
        state, _=env.reset()
        done,trunc, steps, total_reward = False,False, 0, 0
        game+=1
        while not (done or trunc):
            steps += 1
            
            a = model.select_action(state, deterministic=False, with_logprob=False)
            act = Action_adapter(a,max_action)


            s_prime, reward, done, trunc, info = env.step(act)
            reward = Reward_adapter(reward, index)

            if (done or trunc) or (steps > max_steps):
                rewards.append(total_reward)
                mean_score = np.mean(rewards[-100:])

                if total_reward > record:
                    record = total_reward

                dw = True
                guardarPuntuacion(total_reward,nameEnv+'_SAC.csv')
                print("Game: {}, reward {}, best reward {}, mean score {}, saving model...".format(game, total_reward, record,mean_score))
            else:
                dw = False

            replay_buffer.add(state, a, reward, s_prime, dw)
            state = s_prime
            total_reward += reward

            if total_steps >= update_after and total_steps % update_every == 0:
                for j in range(update_every):
                    model.train(replay_buffer)



            if total_steps % eval_interval == 0:
                score = evaluate_policy(eval_env, model, max_steps, max_action)
                print('Name:',nameEnv,'steps: {}k'.format(int(total_steps/1000)),'score:', score)
            
            total_steps += 1

            if game % 500==0:
                model.save('SAC',game,nameEnv)
    env.close()