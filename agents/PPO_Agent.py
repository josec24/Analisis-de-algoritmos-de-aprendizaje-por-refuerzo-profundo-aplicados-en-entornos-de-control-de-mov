import copy
import numpy as np
import torch
import math
from models.PPO_network import BetaActor,Critic

from buffer.batch import Batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

class PPO(object):
	def __init__(self,state_dim,action_dim,env_with_Dead):
		self.actor = BetaActor(state_dim, action_dim, net_width=256).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.critic = Critic(state_dim, net_width=256).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.env_with_Dead = env_with_Dead
		self.action_dim = action_dim
		self.gamma = 0.99
		self.lambd = 0.95
		self.clip_rate = 0.2
		self.K_epochs = 10
		self.data = []
		self.l2_reg = 1e-3
		self.a_optim_batch_size =  64
		self.c_optim_batch_size = 64
		self.entropy_coef = 0
		self.entropy_coef_decay = 0.9998

		self.Batch=Batch(self.env_with_Dead)

	def select_action(self, state):
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			dist = self.actor.get_dist(state)
			act = dist.sample()
			act = torch.clamp(act, 0, 1)
			logprob_a = dist.log_prob(act).cpu().numpy().flatten()
			return act.cpu().numpy().flatten(), logprob_a

	def evaluate(self, state):
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			act = self.actor.dist_mode(state)
			return act.cpu().numpy().flatten(),0.0


	def train(self):
		self.entropy_coef*=self.entropy_coef_decay
		state, act, reward, s_prime, logprob_a, done_mask, dw_mask, self.data = self.Batch.makeBatch(self.data)

		with torch.no_grad():
			vs = self.critic(state)
			vs_ = self.critic(s_prime)

			deltas = reward + self.gamma * vs_ * (1 - dw_mask) - vs

			deltas = deltas.cpu().flatten().numpy()
			adv = [0]

			for dlt, mask in zip(deltas[::-1], done_mask.cpu().flatten().numpy()[::-1]):
				advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - mask)
				adv.append(advantage)
			adv.reverse()
			adv = copy.deepcopy(adv[0:-1])
			adv = torch.tensor(adv).unsqueeze(1).float().to(device)
			td_target = adv + vs
			adv = (adv - adv.mean()) / ((adv.std()+1e-4)) 
		a_optim_iter_num = int(math.ceil(state.shape[0] / self.a_optim_batch_size))
		c_optim_iter_num = int(math.ceil(state.shape[0] / self.c_optim_batch_size))
		for i in range(self.K_epochs):

			perm = np.arange(state.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(device)
			state, act, td_target, adv, logprob_a = \
				state[perm].clone(), act[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

			for i in range(a_optim_iter_num):
				index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, state.shape[0]))
				distribution = self.actor.get_dist(state[index])
				dist_entropy = distribution.entropy().sum(1, keepdim=True)
				logprob_a_now = distribution.log_prob(act[index])
				ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))

				surr1 = ratio * adv[index]
				surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
				a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

				self.actor_optimizer.zero_grad()
				a_loss.mean().backward()
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
				self.actor_optimizer.step()

			for i in range(c_optim_iter_num):
				index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, state.shape[0]))
				c_loss = (self.critic(state[index]) - td_target[index]).pow(2).mean()
				for name,param in self.critic.named_parameters():
					if 'weight' in name:
						c_loss += param.pow(2).sum() * self.l2_reg

				self.critic_optimizer.zero_grad()
				c_loss.backward()
				self.critic_optimizer.step()

	def put_data(self, transition):
		self.data.append(transition)

	def save(self,episode):
		torch.save(self.critic.state_dict(), "./model/ppo_critic{}.pth".format(episode))
		torch.save(self.actor.state_dict(), "./model/ppo_actor{}.pth".format(episode))

	def load(self,episode):
		self.critic.load_state_dict(torch.load("./model/ppo_critic{}.pth".format(episode)))
		self.actor.load_state_dict(torch.load("./model/ppo_actor{}.pth".format(episode)))