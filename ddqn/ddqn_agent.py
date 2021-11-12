from collections import deque

import numpy as np
import torch
import torch.nn as nn

from ddqn.ddqn_net import DuelingDQNet


class DuelingDQNAgent:
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
        self.q_net = DuelingDQNet(image_size, data_size, num_actions)
        self.loss_fn = nn.SmoothL1Loss().to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-3)

        self.network_sync_freq = sync_freq
        self.gamma = torch.tensor(0.95).float()
        self.experience_buffer = deque(maxlen=exp_replay_size)
        self.batch_size = batch_size
        print("Created agent with ", num_actions, "action space")

    def get_action(self, image, data, num_actions, epsilon):
        with torch.no_grad():
            Qp = self.q_net(image, data)
        A = torch.argmax(Qp)
        A = A if torch.rand(1,).item() > epsilon else torch.randint(0, num_actions, (1,))
        return A

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

    def train(self):
        s_i, s_d, a, rn, sn_i, sn_d = self.experience_rollout()

        # predict expected return of current state using main network
        qp = self.q_net(s_i, s_d)
        current = qp.max(1).values.unsqueeze(1)

        # get target return
        with torch.no_grad():
            qn = self.q_net(sn_i, sn_d)
        q_next = qn.max(1).values.unsqueeze(1)

        rn = np.reshape(rn, (len(rn), 1))
        target = rn + self.gamma * q_next

        loss = self.loss_fn(current, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print("[loss]:", round(loss.item(), 5))
        return loss.item()
