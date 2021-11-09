import gym
from torch_actor_critic_discrete import NewAgent
from utils import plotLearning
from gym import wrappers
import numpy as np
from dqn.dqn_environment import MyRaceTrack
import cv2
import torch

def preprocessing(obs, info):
    # convert to grayscale
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    # resize to [40,40]
    obs = cv2.resize(obs, (40, 40), interpolation=cv2.INTER_AREA)
    # add new axis to [1,40,40]
    obs = obs[np.newaxis, :]
    # extract values
    # print(info)

    info = np.array([info["d_center"]/250, info["d_angle"]/180])
    # info = np.array(list(info.values()))

    # print(info)
    # print(info/360)

    obs = obs/ 255
    # info[0]
    # info = info / 360
    # print(info)
    # info = info[3:]  # 1x3
    return obs, info


if __name__ == '__main__':
    deg2rad = np.pi / 180
    steering_step1 = 1 * deg2rad
    steering_step2 = 2 * deg2rad
    steering_step4 = 4 * deg2rad
    steering_step8 = 8 * deg2rad
    action_map = (
        [1, 0, 0],
        # [1, 0, steering_step1],
        # [1, 0, -steering_step1],
        # [1, 0, steering_step2],
        # [1, 0, -steering_step2],
        # [1, 0, steering_step4],
        # [1, 0, -steering_step4],
        [1, 0, steering_step8],
        [1, 0, -steering_step8],
    )
    num_actions = len(action_map)
    image_size = [1, 1, 40, 40]
    data_size = [1, 2]
    num_of_episodes = 10000
    sync_freq = 10
    exp_replay_size = 1000
    batch_size = 32
    count = 0
    eps = 1

    env = MyRaceTrack()

    agent = NewAgent(alpha=0.00001, input_dims=image_size, data_shape=data_size, gamma=0.99, layer1_size=2048, layer2_size=512, n_actions=len(action_map))

    # env = gym.make('LunarLander-v2')
    score_history = []
    score = 0
    num_episodes = 2000
    for i in range(num_episodes):

        #env = wrappers.Monitor(env, "tmp/lunar-lander",
        #                            video_callable=lambda episode_id: True, force=True)
        done = False
        score = 0
        observation, info = env.reset()
        
        observation, info = preprocessing(observation, info)
        observation = observation[np.newaxis, :]
        info = info[np.newaxis, :]


        # print(observation.shape)
        while not done:
            # print('[test]episode: ', i,'score: %.2f' % score)
            p = np.random.random()
            if p > 0.1:
                action = np.random.randint(0, num_actions)
                print("[np]",action)
            else:
                with torch.no_grad():
                    action = agent.choose_action(observation, info)
                    print("[model]",action)

            observation_, reward, done, info_ = env.step(action_map[action])
            observation_, info_ = preprocessing(observation_, info_)
            observation_ = observation_[np.newaxis, :]
            info_ = info_[np.newaxis, :]

            # print(observation.shape, observation_.shape)

            agent.learn(observation, info, reward, observation_, info_, done)
            observation = observation_
            score += reward

        score_history.append(score)
        print('episode: ', i,'score: %.2f' % score)

        # if eps > 0.05:
        #     eps -= 1 / 100

    done = False
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        observation = observation_
            
    filename = 'Lunar-Lander-actor-critic-new-agent-alpha00001-beta00005-2048x512fc-2000games.png'
    plotLearning(score_history, filename=filename, window=50)
