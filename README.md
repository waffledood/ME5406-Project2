# ME5406

Group 60
- Ng Wei Jie, Brandon (A0184893L)
- Mohamad Haikal Bin Mohamad Yusuf (A0182446B) 
- Dennis Goh Wen Qin (A0096927L)


## Setup 
- Develop and test on Python3.6 in Ubuntu 18
```
sudo apt install python3-pip
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
```
- Setup virtualenv
```
python3 -m virtualenv env
source env/bin/activate
```
- Install python3 libraries
```
pip3 install -r requirements.txt
```

## Train RL Models
- Train DQN
```
is_eval=0 python3 main_dqn.py
```
- Train DDQN
```
is_eval=0 python3 main_ddqn.py
```
- Train A2C
```
is_eval=0 python3 main_a2c.py
```

## Evaluate RL Models
- Evaluate DQN (model is stored as `models/best_dqn.pt`)
```
is_eval=1 python3 main_dqn.py
```
- Evaluate DDQN (model is stored as `models/best_ddqn.pt`)
```
is_eval=1 python3 main_ddqn.py
```
- Evaluate A2C (model is stored as `models/best_a2c.pt`)
```
is_eval=1 python3 main_a2c.py
```

## Simulator
- Control simulator with keyboard
```
python3 demo.py
```

- Observation Space
<p align="center">
  <img alt="screen_debug" src="imgs/obs_space_still.png" width="25%">
  <img alt="screen" src="imgs/obs_space_tilt.png" width="25%">
</p>
- Game Engine
<p align="center">
  <img alt="obs_space" src="imgs/game_with_debug_off.png" width="25%">
  <img alt="obs_space" src="imgs/game_with_debug_on.png" width="25%">
</p>

## Performance
- DQN
<p align="center">
  <img alt="screen_debug" src="imgs/episode_length_dqn.jpg" width="25%">
  <img alt="screen" src="imgs/reward_dqn.jpg" width="25%">
  <img alt="screen" src="imgs/losses_dqn.jpg" width="25%">
</p>

- DDQN
<p align="center">
  <img alt="screen_debug" src="imgs/episode_length_ddqn.jpg" width="25%">
  <img alt="screen" src="imgs/reward_ddqn.jpg" width="25%">
  <img alt="screen" src="imgs/losses_ddqn.jpg" width="25%">
</p>

- A2C
<p align="center">
  <img alt="screen_debug" src="imgs/episode_length_a2c.jpg" width="25%">
  <img alt="screen" src="imgs/reward_a2c.jpg" width="25%">
</p>
<p align="center">
  <img alt="screen" src="imgs/entropy_losses_a2c.jpg" width="25%">
  <img alt="screen" src="imgs/value_losses_a2c.jpg" width="25%">
  <img alt="screen" src="imgs/policy_losses_a2c.jpg" width="25%">
</p>
