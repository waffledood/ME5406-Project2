import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from DuelingDQNAgent import DuelingDQNAgent
from duelingDQN_environment import MyRaceTrack


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
    epsilon = 1
    count = 0
    env = MyRaceTrack()
    agent = DuelingDQNAgent(
        env,
        num_of_episodes,
        sync_freq,
        exp_replay_size,
        batch_size,
        num_actions,
        image_size,
        data_size,
    )
    losses_list, reward_list, episode_len_list = [], [], []
    count = 0

    try:
        # training
        for i in range(5000):
            obs, info = env.reset()
            obs, info = preprocessing(obs, info)
            s = (obs, info)
            obs = obs[np.newaxis, :]
            info = info[np.newaxis, :]
            done = False
            losses, ep_len, rew = 0, 0, 0
            while done != True and ep_len < 2000:
                ep_len += 1
                # get best action
                with torch.no_grad():
                    a = agent.get_action(obs, info, num_actions, epsilon)
                obs, reward, done, info = env.step(action_map[a])
                obs, info = preprocessing(obs, info)
                sn = (obs, info)
                obs = obs[np.newaxis, :]
                info = info[np.newaxis, :]
                agent.collect_experience([s, a, reward, sn])
                s = sn
                count = count + 1
                rew += reward
                if count > 20:
                    count = 0
                    for j in range(4):
                        loss = agent.train()
                        losses += loss
            if epsilon > 0.01:
                epsilon -= 1 / 100
            losses_list.append(losses / ep_len), reward_list.append(rew), episode_len_list.append(
                ep_len
            )
            if i % 20 == 0:
                print(i)
    except KeyboardInterrupt:
        print("Interrupt training")

    torch.save(agent.q_net.state_dict(), "dqn_model.pt")
    plt.plot(episode_len_list)
    plt.xlabel("number of episodes")
    plt.ylabel("episode length")
    plt.show()
    plt.plot(reward_list)
    plt.xlabel("number of episodes")
    plt.ylabel("reward")
    plt.show()
    plt.plot(losses_list)
    plt.xlabel("number of episodes")
    plt.ylabel("losses")
    plt.show()
