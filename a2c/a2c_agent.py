import cv2
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import numpy as np
from a2c.a2c_net import A2CNet
from collections import deque
import copy
import random

class A2CAgent:
	def __init__(self, env, num_episodes, beta, gamma, clip_grad, batch_size, num_actions, image_size, data_size):
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.net = A2CNet(image_size, data_size, num_actions)
		self.loss_fn = nn.SmoothL1Loss().to(self.device)
		self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3) #alpha = 1e-3
		
		self.beta = torch.tensor(beta).float()
		self.gamma = torch.tensor(gamma).float()
		self.clip_grad = torch.tensor(clip_grad).float()
		self.batch_size = batch_size
		self.experience_buffer = deque(maxlen=self.batch_size)
		print('Created agent with ', num_actions, 'action space')
		return
		
	def get_action(self, image, data):
		# We do not require gradient at this point, because this function will be used either
		# during experience collection or during inference
		with torch.no_grad():
			mu, var, _ = self.net(image, data)
		mu = mu.data.to(self.device).cpu().detach().numpy()
		sigma = torch.sqrt(var).data.to(self.device).cpu().detach().numpy()
		actions = np.random.normal(mu, sigma)
		# print(actions)
		actions = np.clip(actions, -0.1, 0.1)
		# print(mu, sigma)
		return actions.squeeze(0)
		
	def calc_logprob(self, mu, var, actions):
		p1 = - ((mu - actions) ** 2) / (2*var.clamp(min=1e-3))
		p2 = - torch.log(torch.sqrt(2 * np.pi * var))
		#print (p1+p2)
		return p1 + p2
	
	def collect_experience(self, experience):
		self.experience_buffer.append(experience)
		return
	
	def experience_rollout(self):
		s_i = torch.tensor([exp[0][0] for exp in self.experience_buffer]).float()
		s_d = torch.tensor([exp[0][1] for exp in self.experience_buffer]).float()
		a = torch.tensor([exp[1] for exp in self.experience_buffer]).float()
		rn = torch.tensor([exp[2] for exp in self.experience_buffer]).float()
		sn_i = torch.tensor([exp[3][0] for exp in self.experience_buffer]).float()
		sn_d = torch.tensor([exp[3][1] for exp in self.experience_buffer]).float()
		return s_i, s_d, a, rn, sn_i, sn_d
	
	def train(self, done):
		s_i, s_d, a, rn, sn_i, sn_d = self.experience_rollout()
		
		# calculate discounted returns and advantages
		returns = np.reshape(rn, (len(rn), 1))
		s_mu, s_var, s_val = self.net(s_i, s_d)
		with torch.no_grad():
			_, _, sn_val = self.net(sn_i, sn_d)
		sn_val = sn_val.data.squeeze(0)
		if done == True:
			sn_val[-1] = 0
		target = returns + self.gamma * sn_val
		#print(target)
		adv = target - s_val.detach()
		
		loss_value = self.loss_fn(target, s_val)
		log_prob = adv * self.calc_logprob(s_mu, s_var, a)
		loss_policy = -log_prob.mean()
		entropy_loss = self.beta * (-(torch.log(2*np.pi*s_var) + 1)/2).mean()
		print(loss_value, loss_policy, entropy_loss)
		
		loss = loss_policy + entropy_loss + loss_value
		self.optimizer.zero_grad()
		loss.backward()  # Compute gradients
		nn_utils.clip_grad_norm_(self.net.parameters(), self.clip_grad)
		self.optimizer.step()  # Backpropagate error
		return loss_policy.item(), loss_value.item(), entropy_loss.item()
		