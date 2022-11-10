import numpy as np
import gym
from agents.PPO_Agent import PPO

def Action_adapter(act,max_action):
    return  2*(act-0.5)*max_action

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
            act, _ = model.evaluate(state)
            act = Action_adapter(act, max_action)
            s_prime, reward, done,trunc, _ = env.step(act)

            total_reward += reward
            steps += 1
            state = s_prime
        scores += total_reward
    return scores/turns

if __name__ == '__main__':
    Envs = ['BipedalWalker-v3','BipedalWalkerHardcore-v3']
    index=0
    
    while(True):
        print('Ingresde el numero para indicar el entorno:\n1)BipedalWalker-v3\n2)BipedalWalkerHardcore-v3')
        index=int(input())
        if index==1 or index==2:
            break

    index-=1      
    nameEnv=Envs[index]
    Env_With_Dead = [True, True]
    Loadmodel=False
    ModelIdex=500 
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

    ppo = PPO(env.observation_space.shape[0],env.action_space.shape[0],Env_With_Dead[index])
    if Loadmodel: ppo.load(ModelIdex)

    traj_lenth = 0
    total_steps = 0
    record=0
    rewards = []
    game=0
    while total_steps < Max_train_steps:
        state, _=env.reset()
        done,trunc, steps, total_reward = False,False, 0, 0
        game+=1
        while not (done or trunc):
            traj_lenth += 1
            steps += 1

            act, logprob_a = ppo.select_action(state)

            act = Action_adapter(act,max_action)
            s_prime, reward, done, trunc, info = env.step(act)
            reward = Reward_adapter(reward, index)

            if (done or trunc) and steps != max_steps:
                rewards.append(total_reward)
                mean_score = np.mean(rewards[-100:])

                if total_reward > record:
                    record = total_reward

                dw = True
                print("Game: {}, reward {}, best reward {}, mean score {}, saving model...".format(game, total_reward, record,mean_score))
            else:
                dw = False

            ppo.put_data((state, act, reward, s_prime, logprob_a, done, dw))
            state = s_prime
            total_reward += reward

            if traj_lenth % T_horizon == 0:
                ppo.train()
                traj_lenth = 0

            if total_steps % eval_interval == 0:
                score = evaluate_policy(eval_env, ppo,max_steps, max_action)
                print('Name:',nameEnv,'steps: {}k'.format(int(total_steps/1000)),'score:', score)
            total_steps += 1

            if total_steps % save_interval==0:
                ppo.save(total_steps)
    env.close()