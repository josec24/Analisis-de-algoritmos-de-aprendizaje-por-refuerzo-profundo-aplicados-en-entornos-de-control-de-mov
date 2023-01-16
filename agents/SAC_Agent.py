import copy
import torch
from models.SAC_network import Actor,Q_Critic
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAC_Agent(object):
	def __init__(self,state_dim,action_dim,):

		self.actor = Actor(state_dim, action_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.q_critic = Q_Critic(state_dim, action_dim).to(device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=3e-4)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		for p in self.q_critic_target.parameters():
			p.requires_grad = False

		self.action_dim = action_dim
		self.gamma = 0.99
		self.tau = 0.005
		self.batch_size = 256

		self.alpha = 0.12
		self.adaptive_alpha = True
		if self.adaptive_alpha:
			self.target_entropy = torch.tensor(-action_dim, dtype=float, requires_grad=True, device=device)
			self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=device)
			self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=3e-4)



	def select_action(self, state, deterministic, with_logprob=False):
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			a, _ = self.actor(state, deterministic, with_logprob)
		return a.cpu().numpy().flatten()



	def train(self,replay_buffer):
		s, a, r, s_prime, dead_mask = replay_buffer.sample(self.batch_size)

		with torch.no_grad():
			a_prime, log_pi_a_prime = self.actor(s_prime)
			target_Q1, target_Q2 = self.q_critic_target(s_prime, a_prime)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = r + (1 - dead_mask) * self.gamma * (target_Q - self.alpha * log_pi_a_prime)

		current_Q1, current_Q2 = self.q_critic(s, a)

		q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		for params in self.q_critic.parameters():
			params.requires_grad = 	False

		a, log_pi_a = self.actor(s)
		current_Q1, current_Q2 = self.q_critic(s, a)
		Q = torch.min(current_Q1, current_Q2)

		a_loss = (self.alpha * log_pi_a - Q).mean()
		self.actor_optimizer.zero_grad()
		a_loss.backward()
		self.actor_optimizer.step()

		for params in self.q_critic.parameters():
			params.requires_grad = 	True

		if self.adaptive_alpha:
			
			alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
			self.alpha_optim.zero_grad()
			alpha_loss.backward()
			self.alpha_optim.step()
			self.alpha = self.log_alpha.exp()

		for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



	def save(self,model_name,episode,name):
		torch.save(self.actor.state_dict(), "./model/{}/{}_sac_actor{}.pth".format(model_name,name,episode))
		torch.save(self.q_critic.state_dict(), "./model/{}/{}_sac_q_critic{}.pth".format(model_name,name,episode))


	def load(self,model_name,episode,name):
		self.actor.load_state_dict(torch.load("./model/{}/{}_sac_actor{}.pth".format(model_name,name,episode)))
		self.q_critic.load_state_dict(torch.load("./model/{}/{}_sac_q_critic{}.pth".format(model_name,name,episode)))









