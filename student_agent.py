import gym
import torch
import torch.nn as nn
from train_my_dbystep_hupdate import DuelingQNet, SkipFrame
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym
import cv2
import numpy as np
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from collections import deque

skip_frames_n = 4
device = torch.device("cpu")
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

action_size = env.action_space.n
state_size = env.observation_space.shape



weight_pth = "weight_2.pth"
q_net = DuelingQNet(action_size).to(device)
state_dict = torch.load(weight_pth, map_location=device)
q_net.load_state_dict(state_dict)
q_net.eval()


# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.skip_count = 0
        self.last_action = 0
        self.initial = True
        self.frames_q = deque(maxlen=4)


    def act(self, observation):

        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action
        
        frame = cv2.resize(cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)
        self.frames_q.append(frame)

        if self.initial:
            self.frames_q.clear()
            for _ in range(4):
                self.frames_q.append(frame)
            self.initial = False

        
        state = np.stack(self.frames_q, axis=0)
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0).div_(255.0)


        with torch.no_grad():
            q_values = q_net(state)
            action = int(q_values.argmax(dim=1).item())

        self.last_action = action
        self.skip_count = skip_frames_n - 1

        return action