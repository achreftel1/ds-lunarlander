import gymnasium as gym
import numpy as np
import torch

env = gym.make("LunarLander-v3")
obs, info = env.reset(seed=0)
print("Gymnasium OK")
print("  observation shape:", obs.shape)          # expect (8,)
print("  action space:", env.action_space)        # expect Discrete(4)
print("Torch OK, version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
env.close()