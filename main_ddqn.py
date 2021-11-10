import json
import os

import cv2
import numpy as np
import torch

from ddqn.ddqn_agent import DuelingDQNAgent
from ddqn.ddqn_environment import MyRaceTrack

is_eval = int(os.environ.get("is_eval"))


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


def train():
    epsilon = 1
    losses_list, reward_list, episode_len_list = [], [], []
    count = 0
    ckpt_idx = 0

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
                    for j in range(3):
                        loss = agent.train()
                        losses += loss
            if epsilon > 0.05:
                epsilon -= 5 / 1000
            losses_list.append(losses / ep_len), reward_list.append(rew), episode_len_list.append(
                ep_len
            )
            print("Episode:", i, "Reward:", rew, "Losses:", losses / ep_len, "Duration:", ep_len)
            torch.save(agent.q_net.state_dict(), f"models/ddqn_{ckpt_idx}.pt")
            ckpt_idx += 1
    except KeyboardInterrupt:
        print("Interrupt training")

    with open("models/ddqn_training_status.json", "w") as f:
        training_status = {
            "episode_length": episode_len_list,
            "reward": reward_list,
            "losses": losses_list,
        }
        json.dump(training_status, f, indent=2)


def test():
    epsilon = 0
    # load model
    # agent.target_net.load_state_dict(torch.load("models/best_ddqn.pt"))
    agent.q_net.load_state_dict(torch.load("models/best_ddqn.pt"))

    # testing
    for i in range(5):
        obs, info = env.reset()
        obs, info = preprocessing(obs, info)
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
            obs = obs[np.newaxis, :]
            info = info[np.newaxis, :]
            rew += reward


if __name__ == "__main__":
    deg2rad = np.pi / 180
    steering_step1 = 1 * deg2rad
    steering_step2 = 2 * deg2rad
    steering_step4 = 4 * deg2rad
    steering_step8 = 8 * deg2rad
    action_map = (
        [1, 0, 0],
        [1, 0, steering_step1],
        [1, 0, -steering_step1],
        [1, 0, steering_step2],
        [1, 0, -steering_step2],
        [1, 0, steering_step4],
        [1, 0, -steering_step4],
        [1, 0, steering_step8],
        [1, 0, -steering_step8],
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
    if is_eval:
        test()
    else:
        train()