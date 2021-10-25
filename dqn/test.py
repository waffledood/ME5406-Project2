import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from DQAgent import DQAgent
from dqn_environment import MyRaceTrack


def preprocessing(obs, info):
    # convert to grayscale
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    # resize to [40,40]
    obs = cv2.resize(obs, (40, 40), interpolation=cv2.INTER_AREA)
    # add new axis to [1,40,40]
    obs = obs[np.newaxis, :]
    # extract values
    info = np.array(list(info.values()))
    info = info[3:]  # 1x3
    return obs, info


if __name__ == "__main__":
    deg2rad = np.pi / 180
    steering_step = 10 * deg2rad
    # 6 actions
    action_map = (
        [1, 0, 0],
        [1, 0, steering_step],
        [1, 0, -steering_step],
    )
    num_actions = len(action_map)
    image_size = [1, 1, 40, 40]
    data_size = [1, 3]
    num_of_episodes = 10000
    sync_freq = 10
    exp_replay_size = 1000
    batch_size = 32
    epsilon = 0
    env = MyRaceTrack()
    agent = DQAgent(
        env,
        num_of_episodes,
        sync_freq,
        exp_replay_size,
        batch_size,
        num_actions,
        image_size,
        data_size,
    )
    reward_list, episode_len_list = [], []

    # load model
    agent.target_net.load_state_dict(torch.load("dqn_model.pt"))
    agent.q_net.load_state_dict(torch.load("dqn_model.pt"))

    # testing
    for i in range(5):
        obs, info = env.reset()
        obs, info = preprocessing(obs, info)
        s = (obs, info)
        obs = obs[np.newaxis, :]
        info = info[np.newaxis, :]
        done = False
        ep_len, rew = 0, 0
        while done != True and ep_len < 2000:
            ep_len += 1
            # get best action
            with torch.no_grad():
                a = agent.get_action(obs, info, num_actions, epsilon)
            obs, reward, done, info = env.step(action_map[a])
            obs, info = preprocessing(obs, info)
            sn = (obs, info)
            s = sn
            obs = obs[np.newaxis, :]
            info = info[np.newaxis, :]
            rew += reward
        print(ep_len)
        reward_list.append(rew), episode_len_list.append(ep_len)

    plt.plot(episode_len_list)
    plt.show()
    plt.plot(reward_list)
    plt.show()
