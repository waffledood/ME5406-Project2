from common.environment3 import Environment
import matplotlib.pyplot as plt
import numpy as np

# init helper variables, 
deg2rad = np.pi/180
steering_step = 10*deg2rad

# init environment 
env = Environment()
num_of_episodes = 10

# init stats 
reward_list, episode_len_list = [], []

# PID parameters
Kp = 0.001
Ki = 0.00000000001
Kd = 0.001 
dt = 0.1

# begin driving
for i in range(num_of_episodes):
    obs, info = env.reset()
    done = False
    ep_len, rew = 0, 0
    a = [1, 0, 0]

    # PID variables 
    P_error_trans, I_error_trans, D_error_trans = 0, 0, 0
    # previous error & correction agent needs to make 
    prev_error_trans, corr_trans = 0, 0

    while(done != True and ep_len < 2000):

        # take an action 
        obs, reward, done, info = env.step(a)
        
        ''' info '''
        velocity = info["velocity"]
        error_trans = info["d_center"]
        error_angle = info["d_angle"]

        ''' debug print '''
        print(f"reward: {reward:.2f}, info: ['velocity': {velocity:.2f}, 'd_center': {error_trans:.2f}, 'd_angle': {error_angle:.2f}]")

        ''' PID Controller '''
        # P error (current error)
        P_error_trans = error_trans
        # I error (cumulative error) 
        I_error_trans = I_error_trans + error_trans
        # D error (difference in errors)
        D_error_trans = (error_trans - prev_error_trans)/dt 

        # PID control algo
        corr_trans = Kp*P_error_trans + Ki*I_error_trans + Kd*D_error_trans

        # updating of variables 
        prev_error_trans = error_trans
        ep_len += 1
        rew  += reward
       
        # determine next action to take 
        a = [1, 0, corr_trans]
    
    # track stats
    reward_list.append(rew), episode_len_list.append(ep_len)

# Episode Length 
plt.plot(episode_len_list)
plt.title("Episode Length")
plt.ylabel("Length of Episode")
plt.xlabel("Episode No.")
plt.show()

# Rewards received 
plt.plot(reward_list)
plt.title("Rewards received")
plt.ylabel("Value of Reward")
plt.xlabel("Episode No.")
plt.show()

env.close()

