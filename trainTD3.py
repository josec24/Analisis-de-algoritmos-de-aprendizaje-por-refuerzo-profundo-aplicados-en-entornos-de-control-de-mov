import numpy as np
import gym
from agents.TD3_Agent import TD3_Agent

from buffer.batch import BatchTD3

from data.helper import guardarPuntuacion

def Reward_adapter(reward, index):
    if index == 0 or index == 1:
        if reward <= -100: reward = -1
    return reward

def evaluate_policy(env, model, render, turns=3):
    scores = 0
    for j in range(turns):

        s,_= env.reset()

        done, ep_r, steps =False, 0, 0

        trunc=False

        while not (done or trunc):
            a = model.select_action(s)
            s_prime, r, done, trunc, info = env.step(a)

            ep_r += r
            steps += 1
            s = s_prime
            if render: env.render()

        scores += ep_r
    return scores / turns

if __name__ == '__main__':
    Envs = ['BipedalWalker-v3','BipedalWalkerHardcore-v3','HalfCheetah-v4','Humanoid-v4']
    index=0
    
    while(True):
        print('Ingresde el numero para indicar el entorno:\n1)BipedalWalker-v3\n2)BipedalWalkerHardcore-v3\n3)HalfCheetah-v4\n4)Humanoid-v4')
        index=int(input())
        if index==1 or index==2 or index==3 or index ==4:
            break

    index-=1      
    nameEnv=Envs[index]
    Env_With_Dead = [True, True, True, True]
    Loadmodel=False
    ModelIdex=500 
    #env = gym.make(nameEnv)
    env = gym.make(nameEnv,render_mode='human')
    eval_env = gym.make(nameEnv)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_steps = env._max_episode_steps
    T_horizon = 2048
    Max_train_steps = 50000000
    save_interval = 1e5
    eval_interval = 2e3


    expl_noise = 0.15
    max_e_steps = env._max_episode_steps
    start_steps = 10*max_e_steps 
    update_after = 2 * max_e_steps 
    update_every=50
    exp_noise=0.15
    noise_decay=0.998

    game=0

    model = TD3_Agent(Env_With_Dead[index],env.observation_space.shape[0],env.action_space.shape[0],max_action)
    if Loadmodel: model.load(nameEnv,3500)


    replay_buffer = BatchTD3(state_dim, action_dim, max_size=int(1e6))


    traj_lenth = 0
    total_steps = 0
    record=0
    rewards = []
    
    while total_steps < Max_train_steps:
        s, _=env.reset()
        done,trunc, steps, total_reward = False,False, 0, 0
        game+=1
        while not (done or trunc):
            steps += 1


            if total_steps < start_steps:
                a = env.action_space.sample()
            else:
                a = (model.select_action(s) + np.random.normal(0, max_action * expl_noise, size=action_dim)
                        ).clip(-max_action, max_action)
            s_prime, r, done, trunc, info = env.step(a)
            r = Reward_adapter(r, index)


            if (done or trunc) or (steps > max_steps):
                rewards.append(total_reward)
                mean_score = np.mean(rewards[-100:])

                if total_reward > record:
                    record = total_reward

                dw = True
                guardarPuntuacion(total_reward,nameEnv+'_TD3.csv')
                print("Game: {}, reward {}, best reward {}, mean score {}, saving model...".format(game, total_reward, record,mean_score))
            else:
                dw = False

            replay_buffer.add(s, a, r, s_prime, dw)
            s = s_prime
            total_reward += r

            if total_steps >= update_after and total_steps % update_every == 0:
                for j in range(update_every):
                    model.train(replay_buffer)

            '''record'''
            if total_steps % eval_interval == 0:
                expl_noise *= noise_decay
                score = evaluate_policy(eval_env, model, False)

                print('EnvName:', Envs[index], 'steps: {}k'.format(int(total_steps/1000)), 'score:', score)
            total_steps += 1

            '''save model'''
            if game % 500 == 0:
                model.save('TD3',game,nameEnv)
    env.close()