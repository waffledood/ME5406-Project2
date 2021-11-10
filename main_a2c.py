import cv2
import numpy as np
import torch

from a2c.a2c_agent import A2CAgent
from common.environment import Environment


def preprocessing(obs, info):
    # convert to grayscale
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    # resize to [40,40]
    obs = cv2.resize(obs, (40, 40), interpolation=cv2.INTER_AREA)
    # add new axis to [1,40,40]
    obs = obs[np.newaxis, :]
    # extract values
    info = np.array(list(info.values()))

    info = info / 360
    obs = obs / 255
    return obs, info


num_actions = 1
image_size = [1, 1, 40, 40]
data_size = [1, 3]
num_of_episodes = 10000
batch_size = 200
beta = 0.001
gamma = 0.95
clip_grad = 0.1
epsilon = 1
count = 0
env = Environment()
agent = A2CAgent(
    env, num_of_episodes, beta, gamma, clip_grad, batch_size, num_actions, image_size, data_size
)
policy_losses_list, value_losses_list, entropy_losses_list, reward_list, episode_len_list = (
    [],
    [],
    [],
    [],
    [],
)

# training
for i in range(5000):
    obs, info = env.reset()
    obs, info = preprocessing(obs, info)
    s = (obs, info)
    obs = obs[np.newaxis, :]
    info = info[np.newaxis, :]
    done = False
    p_loss, v_loss, e_loss, ep_len, rew = 0, 0, 0, 0, 0
    while done != True and ep_len < 2000:
        ep_len += 1
        # get best action
        with torch.no_grad():
            a = agent.get_action(obs, info)
        obs, reward, done, info = env.step([1, 0, a.squeeze(0)])
        obs, info = preprocessing(obs, info)
        sn = (obs, info)
        obs = obs[np.newaxis, :]
        info = info[np.newaxis, :]
        agent.collect_experience([s, a, reward / 100, sn])
        s = sn
        count = count + 1
        rew += reward
        if count > batch_size or done == True:
            count = 0
            pl, vl, el = agent.train(done)
            agent.experience_buffer.clear()
            p_loss += pl
            v_loss += vl
            e_loss += el
    if epsilon > 0.05:
        epsilon -= 5 / 1000
    policy_losses_list.append(p_loss), value_losses_list.append(v_loss)
    entropy_losses_list.append(e_loss),
    reward_list.append(rew), episode_len_list.append(ep_len)
    print("[episode]:", i, "[reward]:", round(rew, 5), "[duration]:", ep_len)
