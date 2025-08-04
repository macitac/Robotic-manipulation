from mani_skill.utils.wrappers import RecordEpisode
import Stack_cube_new
env = MyCustomTask(num_envs=16, render_mode="sensors")
env = RecordEpisode(env, "./videos", save_trajectory=False)
env.reset(seed=0)
for _ in range(10):
    env.step(env.action_space.sample())
env.close()