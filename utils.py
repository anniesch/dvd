import os
import sys
import json
import pickle
import argparse
import torch
import shutil
import glob
import numpy as np
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio


def load_args():
    parser = argparse.ArgumentParser(description='DVD example training')
    parser.add_argument('--resume', '-r', type=int, default=0, help="resume training from a given checkpoint.")
    parser.add_argument('--gpus', '-g', default = str(0), help="GPU ids to use. Please enter a comma separated list")
    parser.add_argument('--use_cuda', default=True, help="to use GPUs")
    parser.add_argument('--num_tasks', type=int, default=2, help='number of tasks')
    parser.add_argument('--human_data_dir', type=str, help='dir to human data', required=True) #'/iris/u/asc8/workspace/humans/Humans/20bn-something-something-v2-all-videos/'
    parser.add_argument('--sim_dir', type=str, default='demos/', help='dir to sim data')
    parser.add_argument('--root', type=str, default='./', help='root dir') 
    parser.add_argument('--log_dir', type=str, default='trained_models/', help='log directory')
    parser.add_argument('--log_freq', type=int, default=1, help='freq of logging for val set')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--im_size', type=int, default=120, help='size of random crops of images')
    parser.add_argument('--num_epochs', type=int, default=150, help='total number of epochs to train for')
    parser.add_argument('--hidden_size', type=int, default=512, help='latent encoding size')
    parser.add_argument('--batch_size', type=int, default=24, help='10 for w/ robot, 20 for just human w/ 72-length trajs, 40 for 10-length trajs')
    parser.add_argument('--traj_length', type=int, default=0, help='length of sequence to train on, 0 means random between 20-40')
    parser.add_argument('--add_demos', type=int, default=0, help='how many robot demos to add (of each of the several tasks)')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate to begin with')
    parser.add_argument('--pretrained', action='store_true', default=False, help='using pretrained sth sth encoder')
    parser.add_argument('--pretrained_dir', type=str, default='pretrained/video_encoder/') 
    parser.add_argument('--demo_type', type=str, default='both', help='for robot demos, use true=ground truth demos, sv2p=sv2p demos or both=for both types')
    parser.add_argument('--just_robot', action='store_true', default=False, help='train on only robot demos')
    parser.add_argument('--robot_tasks', nargs='*', default=[5, 41, 46, 93, 94], help='if using robot demos, which tasks to include robot demos for')
    parser.add_argument('--human_tasks', nargs='*', default=[5, 41, 44, 46, 93, 94], help='if using robot demos, which tasks to include robot demos for')
    parser.add_argument('--similarity', action='store_true', default=False, help='whether to use similarity discriminator')
    parser.add_argument('--demo_batch_val', type=float, default=0.5, help='if using robot demos during sim discriminator training, then value for batching')
    parser.add_argument('--action_dim', type=int, default=5, help='action dim, only used for behavioral cloning baseline (5 for sim, 4 for widowx)')
    
    args = parser.parse_args()
    args.im_size_x = int(args.im_size * 1.5)
    args.json_data_train = args.root + "something-something-v2-train.json"
    args.json_data_val = args.root + "something-something-v2-validation.json"
    args.json_data_test = args.root + "something-something-v2-test.json"
    args.json_file_labels = args.root + "something-something-v2-labels.json"
    random.seed(args.seed)
    print(args)
    return args

# python train.py --num_tasks 6 --traj_length 0 --log_dir "test/" --similarity --batch_size 24 --im_size 120 --seed 1 --lr 0.01 --pretrained --human_tasks 5 41 44 46 93 94 --human_data_dir /iris/u/asc8/workspace/humans/Humans/20bn-something-something-v2-all-videos/ --sim_dir /iris/u/asc8/workspace/humans/domain-adversarial/dataset/sim/


def remove_module_from_checkpoint_state_dict(state_dict):
    """
    Removes the prefix `module` from weight names that gets added by
    torch.nn.DataParallel()
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def load_json_config(path):
    """ loads a json config file"""
    with open(path) as data_file:
        config = json.load(data_file)
        config = config_init(config)
    return config


def setup_cuda_devices(args):
    device_ids = []
    device = torch.device("cuda" if args.use_cuda else "cpu")
    if device.type == "cuda":
        device_ids = [int(i) for i in args.gpus.split(',')]
    return device, device_ids


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    filename = str(state['epoch']) + filename
    checkpoint_path = os.path.join(save_dir, filename)
    model_path = os.path.join(save_dir, 'model_best.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        print(" > Best model found at this epoch. Saving ...")
        shutil.copyfile(checkpoint_path, model_path)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
