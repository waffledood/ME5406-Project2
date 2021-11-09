import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class ActorCriticNetwork(nn.Module):
    def __init__(self, alpha, input_dims, data_shape, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = 100
        self.fc1_dims = 100
        self.fc2_dims = 100
        self.n_actions = n_actions
        # print(input_dims)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_dims[1], out_channels=32, kernel_size=4, stride=2, padding=1
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
        self.info_fc = nn.Sequential(
            nn.Linear(data_shape[1], n_actions), nn.ReLU(), nn.Flatten()  # 1x12
        )
        self.fc1 = nn.Linear(131, 64)
        self.fc2 = nn.Linear(64, 32)
        self.pi = nn.Linear(32, n_actions)
        self.v = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation, data):
        state = T.Tensor(observation).to(self.device)
        info = T.Tensor(data).to(self.device)
        x = F.relu(self.conv(state))
        y = F.relu(self.info_fc(info))
        x = torch.cat((x, y), 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)
        return (pi, v)

class NewAgent(object):
    """ Agent class for use with a single actor critic network that shares
        the lowest layers. For use with more complex environments such as
        the discrete lunar lander
    """
    def __init__(self, alpha, input_dims, data_shape, gamma=0.99, layer1_size=256, layer2_size=256, n_actions=2):
        self.gamma = gamma
        self.actor_critic = ActorCriticNetwork(alpha, input_dims, data_shape, n_actions=n_actions)

        self.log_probs = None

    def choose_action(self, observation, info):
        # print(observation.shape)
        probabilities, _ = self.actor_critic.forward(observation, info)
        probabilities = F.softmax(probabilities)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.log_probs = log_probs

        return action.item()

    def learn(self, state, info, reward, new_state, info_, done):
        self.actor_critic.optimizer.zero_grad()

        _, critic_value_ = self.actor_critic.forward(new_state, info_)
        _, critic_value = self.actor_critic.forward(state, info)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor_critic.optimizer.step()
