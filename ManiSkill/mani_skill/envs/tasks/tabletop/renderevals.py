#!/usr/bin/env python
import time
import gymnasium as gym
import pick_cube_new

def main():
    # Create the environment (this uses the registered "PickCube-v1" environment)
    env = gym.make("PickCube-v1-new",robot_uids="panda", render_mode="human")
    
    # Reset the environment to initialize the scene
    env.reset()
    
    print("Rendering the environment. Press Ctrl+C to exit.")

    try:
        # Main loop: repeatedly render the environment
        while True:
            env.render()   # This will update the viewer window
            time.sleep(1 / 30)  # Roughly 30 FPS
    except KeyboardInterrupt:
        print("Exiting rendering loop.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
