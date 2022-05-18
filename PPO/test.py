import os
import glob
import time
from datetime import datetime
from math import floor
import torch
import numpy as np
import random
import gym
from metadrive import MetaDriveEnv
from PPO import PPO
def choose_steering(action_index):
    steering_index = floor(action_index / 3)
    switch = {0: -0.5,
              1: 0.0,
              2: 0.5, }
    steering = switch.get(steering_index)
    return steering


def choose_acceleration(action_index):
    acceleration_index = floor(action_index % 3)
    switch = {0: -0.5,
              1: 0.0,
              2: 0.5, }
    acceleration = switch.get(acceleration_index)
    return acceleration
config = dict(
        use_render=True,
        manual_control=False,
        traffic_density=0.0,
        random_agent_model=False,
        random_lane_width=False,
        random_lane_num=False,
        use_lateral=True,
        # map="SCrROXT",
        # map="OCXT",
        start_seed=random.randint(0, 1000),
        map=7,  # seven block
        environment_num=100,
    )

#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    env_name = "MetaDrive"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    # env_name = "RoboschoolWalker2d-v1"
    env = MetaDriveEnv(config)
    has_continuous_action_space = False
    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    env = MetaDriveEnv(config)
    # env = gym.make(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = 9

    # initialize a PPO agent
    is_Train = False
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, is_Train, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained1" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            steering = choose_steering(action)
            acceleration = choose_acceleration(action)
            action = np.array([steering, acceleration])
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    test()
