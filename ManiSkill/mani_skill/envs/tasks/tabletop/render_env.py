import time
import cv2
import gymnasium as gym
import numpy as np
import pick_cube_new
import Stack_cube_new



env = gym.make("StackCube-v1-new", render_mode="human")

obs, info = env.reset()

for _ in range(1000):
    action = np.zeros(env.action_space.shape)  # Replace with your control logic.
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        break

env.close()