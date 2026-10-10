LunarLander-v3 trajectory datasets
==================================

Each trajectory is a matrix of shape (8, K_i):
  rows    = state  x_k = (x, y, vx, vy, angle, ang_vel, leg1, leg2)
  columns = time steps k = 1..K_i   (K_i varies: horizon depends on actions)

Labels are the environment's own terminal reward:
  +100 -> safe landing (correct), -100 -> crash (wrong).

Good-agent landing rate over 50 check episodes: 76%

WRONG trajectories (random agent, crashes only)
  trajectories kept : 10000   each of shape (8, K_i)
  K_i (columns)     : mean 92.8, min 53, max 179
  episodes run      : 10004  (landed 0, crashed 10000, truncated 4)

CORRECT trajectories (PPO (stable-baselines3), model results\ppo_lunarlander.zip, landings only)
  trajectories kept : 10000   each of shape (8, K_i)
  K_i (columns)     : mean 319.3, min 255, max 999
  episodes run      : 11556  (landed 10000, crashed 916, truncated 640)

Load in Python:
  import numpy as np
  d = np.load('correct_trajectories.npz', allow_pickle=True)
  X = d['trajectories']        # object array, X[i].shape == (8, K_i)
  K = d['lengths']             # K_i per trajectory
  print(d['state_names'])      # row labels
