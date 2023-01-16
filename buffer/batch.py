import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Batch:
    def __init__(self,env_with_Dead):
        self.env_with_Dead=env_with_Dead

    def makeBatch(self,data):
        s_lst, a_lst, r_lst, s_prime_lst, logprob_a_lst, done_lst, dw_lst = [], [], [], [], [], [], []
        for transition in data:
            state,act,reward,s_prime,logprob_a,done,dw=transition

            s_lst.append(state)
            a_lst.append(act)
            logprob_a_lst.append(logprob_a)
            r_lst.append([reward])
            s_prime_lst.append(s_prime)
            done_lst.append([done])
            dw_lst.append([dw])

        if not self.env_with_Dead:
            dw_lst = (np.array(dw_lst)*False).tolist()

        data = [] 

        with torch.no_grad():
            state= torch.tensor(np.array(s_lst), dtype=torch.float).to(device)
            act=torch.tensor(np.array(a_lst), dtype=torch.float).to(device)
            reward=torch.tensor(np.array(r_lst), dtype=torch.float).to(device)
            s_prime=torch.tensor(np.array(s_prime_lst), dtype=torch.float).to(device)
            logprob_a=torch.tensor(np.array(logprob_a_lst), dtype=torch.float).to(device)
            done_mask=torch.tensor(np.array(done_lst), dtype=torch.float).to(device)
            dw_mask=torch.tensor(np.array(dw_lst), dtype=torch.float).to(device)

        return state, act, reward, s_prime, logprob_a, done_mask, dw_mask, data


class BatchTD3(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.reward = np.zeros((max_size, 1))
		self.next_state = np.zeros((max_size, state_dim))
		self.dead = np.zeros((max_size, 1))

		self.device = device


	def add(self, state, action, reward, next_state, dead):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		self.dead[self.ptr] = dead

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.dead[ind]).to(self.device)
		)


class RandomBuffer(object):
	def __init__(self, state_dim, action_dim, Env_with_dead , max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.Env_with_dead = Env_with_dead

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.reward = np.zeros((max_size, 1))
		self.next_state = np.zeros((max_size, state_dim))
		self.dead = np.zeros((max_size, 1),dtype=np.uint8)

		self.device = device


	def add(self, state, action, reward, next_state, dead):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state

		if self.Env_with_dead:
			self.dead[self.ptr] = dead
		else:
			self.dead[self.ptr] = False

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		with torch.no_grad():
			return (
				torch.FloatTensor(self.state[ind]).to(self.device),
				torch.FloatTensor(self.action[ind]).to(self.device),
				torch.FloatTensor(self.reward[ind]).to(self.device),
				torch.FloatTensor(self.next_state[ind]).to(self.device),
				torch.FloatTensor(self.dead[ind]).to(self.device)
			)