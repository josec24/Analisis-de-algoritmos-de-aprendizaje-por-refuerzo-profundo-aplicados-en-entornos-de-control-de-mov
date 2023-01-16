import torch
from models.TD3_network import Actor,Q_Critic
from buffer.batch import BatchTD3
import copy
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3_Agent(object):
	def __init__(self, env_with_dw, state_dim, action_dim, max_action):

		self.net_width=128
		self.gamma=0.99

		self.actor = Actor(state_dim, action_dim, self.net_width, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
		self.actor_target = copy.deepcopy(self.actor)

		self.q_critic = Q_Critic(state_dim, action_dim, self.net_width).to(device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=1e-4)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		self.batch_size = 256
		
		self.env_with_dw = env_with_dw
		self.action_dim = action_dim
		self.max_action = max_action
		self.policy_noise = 0.2*max_action
		self.noise_clip = 0.5*max_action
		self.tau = 0.005
		self.delay_counter = -1
		self.policy_delay_freq = 1

	def select_action(self, state):

		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			a = self.actor(state)
		return a.cpu().numpy().flatten()

	def train(self,replay_buffer):
		self.delay_counter += 1
		with torch.no_grad():
			s, a, r, s_prime, dw_mask = replay_buffer.sample(self.batch_size)
			noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
			smoothed_target_a = (
					self.actor_target(s_prime) + noise
			).clamp(-self.max_action, self.max_action)

		target_Q1, target_Q2 = self.q_critic_target(s_prime, smoothed_target_a)
		target_Q = torch.min(target_Q1, target_Q2)

		if self.env_with_dw:
			target_Q = r + (1 - dw_mask) * self.gamma * target_Q 
		else:
			target_Q = r + self.gamma * target_Q

		current_Q1, current_Q2 = self.q_critic(s, a)

		q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		if self.delay_counter == self.policy_delay_freq:
			a_loss = -self.q_critic.Q1(s,self.actor(s)).mean()
			self.actor_optimizer.zero_grad()
			a_loss.backward()
			self.actor_optimizer.step()

			for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			self.delay_counter = -1


	def save(self,model_name,episode,name):
		torch.save(self.actor.state_dict(), "./model/{}/{}_actor{}.pth".format(model_name,episode,name))
		torch.save(self.q_critic.state_dict(), "./model/{}/{}_q_critic{}.pth".format(model_name,episode,name))


	def load(self,EnvName,episode):
		print(EnvName)
		print(episode)
		self.actor.load_state_dict(torch.load("./model/TD3/{}_actor{}.pth".format(EnvName,episode)))
		self.q_critic.load_state_dict(torch.load("./model/TD3/{}_q_critic{}.pth".format(EnvName,episode)))

