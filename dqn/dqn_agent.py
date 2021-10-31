import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from dqn.dqn_net import DQNet


class DQAgent:
    def __init__(
        self,
        env,
        num_episodes,
        sync_freq,
        exp_replay_size,
        batch_size,
        num_actions,
        image_size,
        data_size,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.q_net = DQNet(image_size, data_size, num_actions)
        self.target_net = copy.deepcopy(self.q_net)
        self.loss_fn = nn.SmoothL1Loss().to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-3)

        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.gamma = torch.tensor(0.95).float()
        self.experience_replay = deque(maxlen=exp_replay_size)
        self.batch_size = batch_size
        print("Created agent with ", num_actions, "action space")
        return

    def get_action(self, image, data, num_actions, epsilon):
        # We do not require gradient at this point, because this function will be used either
        # during experience collection or during inference
        with torch.no_grad():
            Qp = self.q_net(image, data)
        A = torch.argmax(Qp)
        A = A if torch.rand(1,).item() > epsilon else torch.randint(0, num_actions, (1,))
        return A

    def get_q_next(self, image, data):
        with torch.no_grad():
            qp = self.target_net(image, data)
        q = qp.max(1).values.unsqueeze(1)
        return q

    def collect_experience(self, experience):
        self.experience_replay.append(experience)
        return

    def sample_from_experience(self, sample_size):
        if len(self.experience_replay) < sample_size:
            sample_size = len(self.experience_replay)
        sample = random.sample(self.experience_replay, sample_size)
        s_i = torch.tensor([exp[0][0] for exp in sample]).float()
        s_d = torch.tensor([exp[0][1] for exp in sample]).float()
        a = torch.tensor([exp[1] for exp in sample]).float()
        rn = torch.tensor([exp[2] for exp in sample]).float()
        sn_i = torch.tensor([exp[3][0] for exp in sample]).float()
        sn_d = torch.tensor([exp[3][1] for exp in sample]).float()
        return s_i, s_d, a, rn, sn_i, sn_d

    def train(self):
        s_i, s_d, a, rn, sn_i, sn_d = self.sample_from_experience(sample_size=self.batch_size)
        if self.network_sync_counter == self.network_sync_freq:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.network_sync_counter = 0

        # predict expected return of current state using main network
        qp = self.q_net(s_i, s_d)
        current = qp.max(1).values.unsqueeze(1)

        # get target return using target network
        q_next = self.get_q_next(sn_i, sn_d)
        rn = np.reshape(rn, (len(rn), 1))
        target = rn + self.gamma * q_next

        loss = self.loss_fn(current, target)
        self.optimizer.zero_grad()
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Backpropagate error

        self.network_sync_counter += 1
        return loss.item()
