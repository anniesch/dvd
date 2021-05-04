from PIL import Image
import torch
import numpy as np
import imageio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_obs(args, info):
    hand = np.array([info['hand_x'], info['hand_y'], info['hand_z']])
    mug = np.array([info['mug_x'], info['mug_y'], info['mug_z']])
    mug_quat = np.array([info['mug_quat_x'], info['mug_quat_y'], info['mug_quat_z'], info['mug_quat_w']])
    init_low_dim = np.concatenate([hand, mug, mug_quat, [info['drawer']], [info['coffee_machine']], [info['faucet']]])
    return init_low_dim


def get_args(flags):
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_dim", type=int, default=5)
    parser.add_argument("--replay_buffer_size", type=int, default=100) # only for logging block interaction
    parser.add_argument("--hidden_size", type=int, default=512) #256) # Dimension of latent space if high dim
    parser.add_argument("--im_size", type=int, default=120) #84)
    parser.add_argument("--verbose", type=bool, default=1) # if logging everything from the environment

    parser.add_argument("--num_epochs", type=int, default=100) #2101
    parser.add_argument("--traj_length", type=int, default=20) #10
    parser.add_argument("--num_traj_per_epoch", type=int, default=3) # 5, 50 steps total per epoch/episode
    parser.add_argument("--batch_sz", type=int, default=32)

    parser.add_argument("--_resample", action='store_true', default=False) # refit on the best K trajectories
    parser.add_argument("--random_act_prob", default=0.01, type=float) # take random action while executing actions from the selected trajectory
    parser.add_argument("--use_classifier", type=str, default='use') # If using learned classifier, set to "use", otherwise set to None
    parser.add_argument("--dynamics_var", action='store_true', default=False) # If using dynamics disagreement as a baseline
    # Logging types
    # 0: no logging 
    # 1: no viz env logging
    # 2: all logging
    parser.add_argument("--logging", default=2, type=int)
    
    ## ROBOT FLAGS
    parser.add_argument("--robot", action='store_true', default=False)
    
    # Specific flags for human videos (log dir)
    parser.add_argument("--root", type=str, default='./')
    parser.add_argument("--model_dir", type=str, default='/trained_models/')
    parser.add_argument("--task_num", type=int, default=93) 
    parser.add_argument('--num_tasks', type=int, default=6, help='number of tasks')
    parser.add_argument('--pretrained', action='store_true', default=True, help='use pretrained video encoder from sth sth')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--xml', type=str, default='env1', help='xml file for env, simple or simple_red')
    
    parser.add_argument('--sample_sz', type=int, default=100, help='sample size for cem')
    parser.add_argument('--resample_ratio', type=float, default=0.2, help='resample ratio for cem')
    parser.add_argument('--cem_iters', type=int, default=2, help='number of iterations for cem')
    
    parser.add_argument('--num_demos', type=int, default=0, help='number of demos')
    parser.add_argument('--demo_path', type=str, default='demos/', help='path to demo for one-shot')
    parser.add_argument('--similarity', action='store_true', default=False, help='whether to use similarity discriminator')
    parser.add_argument('--random', type=bool, default=False, help='if planning with random actions')
    
    # For using the SV2P model
    parser.add_argument("--sv2p_root", default='./')
    parser.add_argument("--sv2p_problem")     
    parser.add_argument("--phorizon", type=int, default=20) #planning horizon
    
    args, unparsed = parser.parse_known_args()
    args.p_horizon = args.traj_length
    
    for key, value in flags.__flags.items():
        vars(args)[key] = value.value
    
    args.im_size_x = int(args.im_size * 1.5) if not args.robot else args.im_size
        
    # Build log directory
    logdir = args.root + args.model_dir[:-8] + '/'
        
    logdir += 'sv2p_task' + str(args.task_num)
    logdir += '_' + args.xml
    logdir += '_seed' + str(args.seed)
    
    if args.random: logdir += '_rand'
    else:
        if args.num_demos:
            assert(args.similarity)
            logdir += '_demos' + str(args.num_demos)
            args.demo_path += f'task{args.task_num}/'

        if args.cem_iters != 0:
            logdir += '_cem' + str(args.cem_iters)

    if args.robot:
        logdir += "_ROBOT"

    args.log_dir = logdir + '/'
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    import json
    with open(args.log_dir + 'commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    print("Log directory:", args.log_dir)
    print('Num epochs', args.num_epochs)
    import random
    random.seed(args.seed)
    return args
