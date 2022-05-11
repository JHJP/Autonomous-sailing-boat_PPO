from torch import nn
import torch
import mlagents
from mlagents_envs.environment import UnityEnvironment as UE

from collections import deque
import itertools
import numpy as np
import random

GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=50000
MIN_REPLAY_SIZE=1000
EPSILON_START=1.0
EPSILON_END=0.02
EPSILON_DECAY=10000
TARGET_UPDATE_FREQ=1000

env = UE(file_name='BoatSailing_RL2', seed=1, side_channels=[])

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))
        
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
        )

    def forward(self, x):
        return self.net(x)
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        # every single operation in pytorch expects batch dimension. Since we are not using batched environment, we don't have a batch dimension. so we use unsqueeze(0)
        # to make fake batch dimensoin size of one.
        q_values = self(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action

env = gym.make('CartPole-v0')

replay_buffer = deque(maxlen=BUFFER_SIZE)
reward_buffer = deque([0.0], maxlen=100)

episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)

# Making parameters of online_net and target_net the same.
target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

## Initializing replay buffer
obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()

    new_obs, reward, done, info = env.step(action)
    transition = (obs, action, reward, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    if done:
        obs = env.reset()

## Main training loop
obs = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    random_sample = random.random()

    if random_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)
    
    new_obs, reward, done, info = env.step(action)
    transition = (obs, action, reward, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    episode_reward += reward

    if done:
        obs = env.reset()

        reward_buffer.append(episode_reward)
        episode_reward = 0.0

    # showing cartpole simulation
    if len(reward_buffer) >= 100:
        if np.mean(reward_buffer) >= 195:
            while True:
                action = online_net.act(obs)
                obs, _, done, _=env.step(action)
                env.render()
                if done:
                    env.reset()


    ### start gradient step
    transitions = random.sample(replay_buffer, BATCH_SIZE)

    # pytorch can use nparray faster than python array. So that make nparray for observations, actions, rewards, dones, new_observations
    obses = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rewards = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])

    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

    ## compute targets for loss function
    target_q_values = target_net(new_obses_t)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
    # algorithm from 2015dqn paper    
    targets = rewards_t + GAMMA*(1-dones_t)*max_target_q_values

    ## compute loss
    # output q_values of actions from obses_t 
    q_values = online_net(obses_t)
    # currently predicted q value for the action we took at the original time of the transition
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
    # use hubor loss
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    # gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ## update target networks
    if step%TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())
    
    ## logging
    if step % 1000 == 0:
        print()
        print('Step', step)
        print('Average reward', np.mean(reward_buffer))