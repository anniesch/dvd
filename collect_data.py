from sim_envs.sim_env.tabletop import Tabletop

from replay_buffer.high_dim_replay import ImageBuffer

import pickle
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time

from PIL import Image
import imageio
import cv2
import gtimer as gt
import copy
import json
import importlib
import av
import h5py

def save_im(im, name):
    img = Image.fromarray(im.astype(np.uint8))
    img.save(name)
    
def get_obs(args, info):
    hand = np.array([info['hand_x'], info['hand_y'], info['hand_z']])
    mug = np.array([info['mug_x'], info['mug_y'], info['mug_z']])
    mug_quat = np.array([info['mug_quat_x'], info['mug_quat_y'], info['mug_quat_z'], info['mug_quat_w']])
    init_low_dim = np.concatenate([hand, mug, mug_quat, [info['drawer']], [info['coffee_machine']], [info['faucet']]])
    return init_low_dim
    
def get_random_action_sequence(env, traj_length, sample_sz=1000):
    act_dim = env.action_space.shape[0]
    acts = np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=[sample_sz, traj_length, act_dim])
    acts[:,:,-1] = np.random.randint(2, size=[sample_sz, traj_length])
    return acts

def take_random_trajectory(args, env, obs):
    actions = get_random_action_sequence(env, args.traj_length, sample_sz = 1)
    actions = actions.squeeze(0)
    action_sample = []
    high_dim_sample = []
    high_dim_sample.append(obs)
    for action in actions:
        next_ob, reward, terminal, env_info = env.step(action)
        high_dim_sample.append(next_ob)
        action_sample.append(action)
    return np.array(high_dim_sample), np.array(action_sample), next_ob
    
def get_human_demos(args):
    '''Hard coded human demos'''
    env = Tabletop(
                    log_freq=args.env_log_freq, 
                    filepath=args.log_dir + '/env',
                    xml=args.xml,
                    verbose=args.verbose)
    
    img_buffer = ImageBuffer(args,
                            trajectory_length=args.num_traj_per_epoch*args.traj_length,
                            action_dim=args.action_dim,
                            savedir=args.log_dir,
                            memory_size=args.replay_buffer_size,
                            state_dim=args.im_size,
                            )
    
    env.max_path_length = args.traj_length * args.num_traj_per_epoch
    
    env.initialize()
    total_good = 0
    for eps in range(args.num_epochs):
        eps_low_dim = []
        eps_obs = []
        eps_next = []
        eps_act = []
        start = time.time()

        obs, env_info = env.reset_model()
        init_im = env.get_obs() * 255 
        if eps == 0 and args.verbose:
            save_im(init_im, '{}/init.png'.format(args.log_dir))

        step = 0
        low_dim_state = get_obs(args, env_info)
        very_start = low_dim_state
        eps_low_dim.append(low_dim_state)
        
        '''Hard coded goals
        This will give a demo consisting of 3 trajectories each of length args.traj_length,
        which subsequently move the robot arm to each of the goal positions. 
        '''
        goals = [[-0.17, 0.55, 0], [-0, 0.6, 0], [0.1, 0.6, 0]]
        for i in range(args.num_traj_per_epoch):
            if args.random:
                imgs, actions, obs = take_random_trajectory(args, env, obs)
            else:
                goal = goals[i] + np.random.uniform(-0.04, 0.04, (3,))
                ob = low_dim_state[:3]
                imgs, actions, env_info = env.move_gripper(ob, goal, args.traj_length)

            eps_obs.append(imgs[:-1])
            eps_next.append(imgs[1:])
            eps_act.append(actions)

        low_dim_state = get_obs(args, env_info)
        eps_low_dim.append(low_dim_state)
        
        eps_obs = np.array(eps_obs).transpose((0, 1, 4, 2, 3))
        eps_obs = eps_obs.reshape(-1, args.num_traj_per_epoch * args.traj_length, 3, args.im_size, int(args.im_size*1.5))
        if eps < 2:
            with imageio.get_writer(args.log_dir + '/demo_sample' + str(eps) + '.gif', mode='I') as writer:
                for k, frame in enumerate(eps_obs[0]):
                    img = np.array(frame)
                    img = img.transpose((1, 2, 0)) * 255.0
                    writer.append_data(img.astype('uint8'))
        # Check criteria to make sure demo is successful
        right_to_left = low_dim_state[3:4] - very_start[3:4]
        left_to_right = -low_dim_state[3:4] + very_start[3:4]
        forward = low_dim_state[4:5] - very_start[4:5]
        drawer_move = np.abs(low_dim_state[10] - very_start[10])
        drawer_open = np.abs(low_dim_state[10])
        move_faucet = low_dim_state[12]

        if args.task_num == 5: # close drawer
            criteria = drawer_open < 0.01 and np.abs(right_to_left) < 0.01
        elif args.task_num == 93: # move faucet to right
            criteria = last_obs[12] < -0.05
        elif args.task_num == 94: # move cup right to left
            criteria = left_to_right < -0.05 and drawer_move < 0.03 and move_faucet < 0.01
        elif args.task_num == 45: # Push cup forward
            criteria = abs(right_to_left) < 0.04 and forward > 0.1
        elif args.task_num == 46: # open drawer
            criteria = last_obs[10] < -0.08 and np.abs(right_to_left) < 0.01
        else:
            print("No criteria set")
            assert(False)

        if criteria:
            eps_obs = np.array(eps_obs).reshape(-1, args.num_traj_per_epoch * args.traj_length, 3, args.im_size, int(args.im_size*1.5))
            if eps < 10:
                with imageio.get_writer(args.log_dir + '/trial' + str(eps) + '.gif', mode='I') as writer:
                    for k, frame in enumerate(eps_obs[0]):
                        img = np.array(frame)
                        img = img.transpose((1, 2, 0)) * 255.0
                        writer.append_data(img.astype('uint8'))
            eps_next = np.array(eps_next).transpose((0, 1, 4, 2, 3))
            eps_next = np.array(eps_next).reshape(-1, args.num_traj_per_epoch * args.traj_length, 3, args.im_size, int(args.im_size*1.5))
            eps_act = np.array(eps_act).reshape(-1, args.num_traj_per_epoch * args.traj_length, img_buffer.action_dim)
            img_buffer.add_sample(
                        states=eps_obs,
                        next_states=eps_next,
                        actions=eps_act,
                        )
            total_good += 1
        end = time.time()
        print("===== EPISODE {} TRY {} FINISHED IN {}s =====".format(total_good, eps, end - start))
        if total_good == args.num_epochs:
            assert(False)
                
                
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_dim", type=int, default=5)
    parser.add_argument("--replay_buffer_size", type=int, default=100) 
    parser.add_argument("--im_size", type=int, default=120) 
    parser.add_argument("--log_dir", type=str, default="data")
    parser.add_argument("--env_log_freq", type=int, default=20) 
    parser.add_argument("--verbose", type=bool, default=1) 
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--traj_length", type=int, default=20)
    parser.add_argument("--num_traj_per_epoch", type=int, default=3) 
    parser.add_argument("--random", action='store_true', default=False) # take random actions
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--xml", type=str, default='env1')
    parser.add_argument("--task_num", type=int, default=94) 

    args = parser.parse_args()
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    get_human_demos(args)