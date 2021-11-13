import numpy as np
import torch
import torch.nn as nn


class DuelingDQNet(nn.Module):
    def __init__(self, image_shape, data_shape, n_actions):
        super(DuelingDQNet, self).__init__()
        # H_out = (H_in - K + 2P)/S + 1
        # W_out = (W_in - K + 2P)/S + 1
        # K: Kernel Size
        # S: Stride
        # P: Padding
        # input image: [N, 1, 40, 80]
        # input data: [N, 3]
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=image_shape[1], out_channels=32, kernel_size=4, stride=2, padding=1
            ),  # 32x20x20
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
            ),  # 32x10x10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x5x5
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=0
            ),  # 32x2x2
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc1 = nn.Sequential(nn.Linear(data_shape[1], 12), nn.ReLU(), nn.Flatten())  # 1x12
        conv_out_size = self._get_conv_out(image_shape)
        fc_out_size = self._get_fc_out(data_shape)

        # state value
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size + fc_out_size, 128), nn.ReLU(), nn.Linear(128, 1)
        )

        # action advantages
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size + fc_out_size, 128), nn.ReLU(), nn.Linear(128, n_actions)
        )

        # self.fc2 = nn.Sequential(
        #     nn.Linear(conv_out_size + fc_out_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, n_actions)
        # )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(*shape))
        return int(np.prod(o.size()))

    def _get_fc_out(self, shape):
        o = self.fc1(torch.zeros(*shape))
        return int(np.prod(o.size()))

    def forward(self, image, data):
        i = self.conv(torch.FloatTensor(image))
        d = self.fc1(torch.FloatTensor(data))
        a = torch.cat((i, d), 1)

        values = self.value_stream(a)
        advantages = self.advantage_stream(a)
        qValues = values + (advantages - advantages.mean())

        return qValues
