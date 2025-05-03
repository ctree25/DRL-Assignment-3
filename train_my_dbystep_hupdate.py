from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym
import cv2
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation

import os
import matplotlib.pyplot as plt
import imageio
import numpy as np
from IPython.display import Image
import torch
import random
from collections import deque

import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast,GradScaler
from tensordict import TensorDict
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler

"""
1. Rank-based prioritization
V 2. Dueling Network
3. Noisy Network（NoisyLinear）
4.  Multi-step return
5. Distributional RL

softupdate -> hard 不用調tao
Buffer to GPU, 但沒PER
train every 4 or 8 steps
smooth_l1_loss
gradient clipping
warm up
"""

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
    
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=True)   
env = ResizeObservation(env, (84, 84))            
env = FrameStack(env, 4)                          

action_size = env.action_space.n
state_size = env.observation_space.shape  # (4, 84, 84)


import torch
import torch.nn as nn

    
class DuelingQNet(nn.Module):
    def __init__(self, n_actions, hidden=512):
        super().__init__()
        # CNN feature extractor
        self.feature = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # (batch, 4, 84, 84) -> (batch, 32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # -> (batch, 64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> (batch, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),  # -> (batch, 64*7*7)
        )

        self.value = nn.Sequential(
            nn.Linear(64*7*7, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(64*7*7, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        x = x / 255.0
        x = self.feature(x)
        v = self.value(x)
        a = self.advantage(x)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q




class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.9, batch_size=32, tau=0.005, 
                repy_buf_capacity=3000,
                device='cuda'):
        
        self.scaler = GradScaler()

        # TODO: Initialize some parameters, networks, optimizer, replay buffer, etc.
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.tau = tau
        self.epsilon = 1.0

        self.q_net = DuelingQNet(action_size).to(device)
        self.target_net = DuelingQNet(action_size).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())  
        self.target_net.eval()  # target net no need to train

        # PER Buffer
        # self.replay_buffer = RankBasedReplayBuffer(capacity=repy_buf_capacity, alpha=repy_buf_alpha, epsilon=repy_buf_epsilon)
        storage = LazyTensorStorage(max_size=repy_buf_capacity, device=device)
        self.replay_buffer = TensorDictReplayBuffer(
            storage=storage,
            sampler=RandomSampler(),
            batch_size=batch_size,
        )

        # Hyperparams
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_step_count = 0
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def get_action(self, state, deterministic=True):

        if not deterministic and torch.rand(1, device=self.device).item() < self.epsilon:
            return torch.randint(self.action_size, (1,), device=self.device).item()
        else:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0).div_(255.0)
                q_values = self.q_net(state)
                return int(q_values.argmax(1).item())


    def update_target_net(self):
        # target = τ * q_net + (1 - τ) * target
        # for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
        #     target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train(self):
        # TODO: Sample a batch from the replay buffer
        if len(self.replay_buffer) < self.batch_size:
          return

        batch = self.replay_buffer.sample()
        states = batch["state"].float() / 255.0
        actions = batch["action"].unsqueeze(-1)
        rewards = batch["reward"]
        next_states = batch["next_state"].float() / 255.0
        dones = batch["done"].float()

        with autocast("cuda"):

            q_values = self.q_net(states).gather(1, actions).squeeze(-1)
            # target = r + γ max_a' Q_target(s', a')
            with torch.no_grad():
                # Double QDN
                #  learning network for action selection
                next_actions = self.q_net(next_states).argmax(1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(-1)
                target_q = rewards + self.gamma * next_q * (1 - dones)

            td_errors = q_values - target_q

            """smooth_l1_loss"""
            loss = F.smooth_l1_loss(q_values, target_q)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)                 # 先 unscale
            torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        # TODO: Update target network periodically
        # self.update_target_net()


    
def main():  
    torch.backends.cudnn.benchmark = True
    train_record_num = 5
    update_target_step = 15000
    episode = 0
    num_steps = 6 * 1000000
    decay_stage_1 = 2.5 * 1000000 
    decay_epsilon_1 = 0.1
    decay_stage_2 = 3 * 1000000 
    decay_epsilon_2 = 0.001

    epsilon_start = 1.0

    agent = DQNAgent(state_size, action_size, lr=5e-5, gamma=0.99, batch_size=128, tau=6e-5,
                    repy_buf_capacity=60000, 
                    device='cuda') 
    warm_up_step = agent.batch_size*15

    
    save_dir = f"./train_record/train_{train_record_num}/"
    os.makedirs(save_dir, exist_ok=True)
    save_dir_w = f"./train_record/train_{train_record_num}/weight/"
    os.makedirs(save_dir_w, exist_ok=True)
    avg_log  = open(f"{save_dir}avg_log.txt", "w", buffering=1)
    im_log  = open(f"{save_dir}im_log.txt", "w", buffering=1)

    reward_history = [] # Store the total rewards for each episode
    avg_reward_history = []
    best_score = 0

    from tqdm import tqdm
    # TODO: Reset the environment
    state = env.reset()
    done = False
    total_reward = 0
    for step in tqdm(range(num_steps), desc="Training steps"):

        if state.shape[-1] == 1:
            state = np.squeeze(state, axis=-1)
        action = agent.get_action(state, deterministic=False)

        # TODO: Add the experience to the replay buffer and train the agent
        # next_state, reward, terminated, truncated, _ = env.step(action)
        next_state, reward, done, _ = env.step(action)
        if next_state.shape[-1] == 1:
            next_state = np.squeeze(next_state, axis=-1)

        
        # agent.replay_buffer.add(state, action, reward, next_state, done)
        transition = TensorDict({
            "state": torch.tensor(state, dtype=torch.uint8, device=agent.device),
            "action": torch.tensor(action, dtype=torch.long, device=agent.device),
            "reward": torch.tensor(reward, dtype=torch.float32, device=agent.device),
            "next_state": torch.tensor(next_state, dtype=torch.uint8, device=agent.device),
            "done": torch.tensor(done, dtype=torch.bool, device=agent.device),
        }, batch_size=[])
        agent.replay_buffer.add(transition)


        if (step+1) % 4 == 0 and len(agent.replay_buffer) > warm_up_step:
            agent.train()
            agent.train_step_count += 1
        
        # TODO: Update the state and total reward
        state = next_state
        total_reward += reward
        if step % update_target_step == 0:
            agent.update_target_net()

        """ epsilon decay
        """
        if step <= decay_stage_1:
            agent.epsilon = max(decay_epsilon_1, epsilon_start - step * (epsilon_start - decay_epsilon_1) / decay_stage_1)
        elif step <= decay_stage_2:
            agent.epsilon = max(decay_epsilon_2, decay_epsilon_1 - (step-decay_stage_1) * (decay_epsilon_1 - decay_epsilon_2) / (decay_stage_2-decay_stage_1))
        else:
            agent.epsilon = decay_epsilon_2

        if done:
            episode += 1
            print(f"Episode {episode}, Reward: {total_reward}, step_num: {step}, epsilon: {agent.epsilon}", file=im_log, flush=True)
            reward_history.append(total_reward)
            if episode % 10 == 0 :
                avg_reward = np.mean(reward_history[-10:])
                print(f"Episode {episode} Avg Reward (last 10 episode): {avg_reward}", file=avg_log, flush=True)
                avg_reward_history.append(avg_reward)
            # TODO: Reset the environment
            state = env.reset()
            done = False
            total_reward = 0
            if episode % 100 == 0:
                torch.save(agent.q_net.state_dict(), f"./train_record/train_{train_record_num}/weight/dqn_weights_{episode+1}.pth")
                

        

    torch.save(agent.q_net.state_dict(), f"./train_record/train_{train_record_num}/weight/dqn_weights.pth")

    avg_log.close()
    im_log.close()

    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training History")
    plt.savefig(f"./train_record/train_{train_record_num}/training_reward_plot.png")  
    plt.close()

    plt.plot(avg_reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.title("Avg Training History")
    plt.savefig(f"./train_record/train_{train_record_num}/avg_training_reward_plot.png")  
    plt.close()


if __name__ == '__main__':
    main()