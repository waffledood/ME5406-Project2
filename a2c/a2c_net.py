import torch
import torch.nn as nn
import numpy as np

class A2CNet(nn.Module):
	def __init__(self, image_shape, data_shape, n_actions):
		super(A2CNet, self).__init__()
		# H_out = (H_in - K + 2P)/S + 1
		# W_out = (W_in - K + 2P)/S + 1
		# K: Kernel Size
		# S: Stride
		# P: Padding
		# input image: [N, 1, 40, 80]
		# input data: [N, 3]
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels=image_shape[1], out_channels=32,
					  kernel_size=4, stride=2, padding=1),  # 32x20x20
			nn.ReLU(),
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4,
					  stride=2, padding=1),  # 32x10x10
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),  # 64x5x5
			nn.Conv2d(in_channels=32, out_channels=32,
					  kernel_size=3, stride=2, padding=0),  # 32x2x2
			nn.ReLU(),
			nn.Flatten()
		)
		self.fc1 = nn.Sequential(
			nn.Linear(data_shape[1], 12),  # 1x12
			nn.ReLU(),
			nn.Flatten()
		)
		conv_out_size = self._get_conv_out(image_shape)
		fc_out_size = self._get_fc_out(data_shape)
		self.fc2 = nn.Sequential(
			nn.Linear(conv_out_size+fc_out_size, 512), # 1x512
			nn.ReLU()
		)
		self.mu = nn.Sequential(
			nn.Linear(512, n_actions),
			nn.Tanh(),
		)
		self.var = nn.Sequential(
			nn.Linear(512, n_actions),
			nn.Softplus(),
		)
		self.value = nn.Linear(512, 1)
	
	def _get_conv_out(self, shape):
		o = self.conv(torch.zeros(*shape))
		return int(np.prod(o.size()))

	def _get_fc_out(self, shape):
		o = self.fc1(torch.zeros(*shape))
		return int(np.prod(o.size()))
		
	def forward(self, image, data):
		i = self.conv(torch.FloatTensor(image))
		d = self.fc1(torch.FloatTensor(data))
		s = torch.cat((i, d), 1)
		f_out = self.fc2(s)
		return self.mu(f_out), self.var(f_out)+0.001, self.value(f_out)

		