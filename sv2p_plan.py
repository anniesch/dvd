from multi_column import MultiColumn, SimilarityDiscriminator
from utils import remove_module_from_checkpoint_state_dict
from plan_utils import get_args, get_obs
from transforms_video import *

from sim_env.tabletop import Tabletop

import pickle
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import torch.optim as optim
from torch import nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import random
import time

from PIL import Image, ImageSequence
import imageio
import cv2
import gtimer as gt
import copy
import json
import importlib
import av
import copy

import sys
from absl import flags, app

from tensor2tensor.bin.t2t_decoder import create_hparams
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


## CONSTANTS ##
TOP_K = 5 # uniformly choose from the top K trajectories
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS = flags.FLAGS

flags.DEFINE_integer('action_dim', 5,'action_dim')
flags.DEFINE_integer('replay_buffer_size', 100,'replay_buffer_size')
flags.DEFINE_integer('hidden_size', 512,'hidden_size')
flags.DEFINE_integer('im_size', 120, 'im_size')
flags.DEFINE_integer('env_log_freq', 10, 'env_log_freq')
flags.DEFINE_integer('verbose', 1, 'verbose')
flags.DEFINE_integer('num_epochs', 100,'num_epochs')
flags.DEFINE_integer('sv2p_epoch', 200001,'sv2p_epoch')
flags.DEFINE_integer('traj_length', 20,'traj_length')
flags.DEFINE_integer('num_traj_per_epoch', 3, 'num_traj_per_epoch')
flags.DEFINE_integer('batch_sz', 32, 'batch_sz')
flags.DEFINE_bool('_resample', False, '_resample')
flags.DEFINE_float('random_act_prob', 0.01, 'random_act_prob')
flags.DEFINE_integer('grad_steps_per_update', 20,'grad_steps_per_update')
flags.DEFINE_bool('dynamics_var', False, 'dynamics_var')
flags.DEFINE_integer('logging', 2,'logging')
flags.DEFINE_string('root', './','root')
flags.DEFINE_string('model_dir', 'trained_models/','model_dir')
flags.DEFINE_integer('task_num', 5,'task_num')
flags.DEFINE_integer('num_tasks', 6,'num_tasks')
flags.DEFINE_integer('seed', 120, 'seed')
flags.DEFINE_string('xml', 'updated','xml')
flags.DEFINE_bool('pretrained', True, 'pretrained')
flags.DEFINE_bool('sanity_check', False, 'sanity_check')
flags.DEFINE_integer('sample_sz', 100,'sample_sz')
flags.DEFINE_float('resample_ratio', 0.2,'resample_ratio')
flags.DEFINE_integer('cem_iters', 2,'cem_iters')
flags.DEFINE_integer('num_demos', 3, 'num_demos')
flags.DEFINE_string('demo_path', 'demos/','demo_path')
flags.DEFINE_bool('similarity', False, 'similarity')
flags.DEFINE_string('sv2p_root', 'tensor2tensor/','sv2p_root')
flags.DEFINE_string('sv2p_problem', 'human_updated','problem')
flags.DEFINE_integer('phorizon', 20, 'phorizon')
flags.DEFINE_integer('robot', 0, 'robot')
flags.DEFINE_bool('random', False, 'if planning with random actions')

def save_im(im, name):
    img = Image.fromarray(im.astype(np.uint8))
    img.save(name)
    
class CEM(object):
    def __init__(self, args, savedir, phorizon,
               cem_samples, cem_iters, cost='similarity'):
        
        self.eps = 0
        self.savedir = savedir
        self.planstep = 0
        self.phorizon = phorizon
        self.cem_samples = cem_samples
        self.cem_iters = cem_iters
        self.verbose = False
        self.num_acts = args.action_dim
        self.cost = cost

        # LOADING SV2P 
        FLAGS.data_dir = args.sv2p_root + 'data/'
        epoch = args.sv2p_epoch
        self.sv2p_model_dir = args.sv2p_root + f'out/model.ckpt-{epoch}'

        FLAGS.problem = args.sv2p_problem
        if args.robot: FLAGS.problem = 'human_widowx'
        FLAGS.hparams = 'video_num_input_frames=5,video_num_target_frames=15'
        FLAGS.hparams_set = 'next_frame_sv2p'
        FLAGS.model = 'next_frame_sv2p'
        
        # Create hparams
        hparams = create_hparams()
        hparams.video_num_input_frames = 1
        hparams.video_num_target_frames = self.phorizon

        # Params
        num_replicas = self.cem_samples
        frame_shape = hparams.problem.frame_shape
        forward_graph = tf.Graph()
        with forward_graph.as_default():
            self.forward_sess = tf.Session()
            input_size = [num_replicas, hparams.video_num_input_frames]
            target_size = [num_replicas, hparams.video_num_target_frames]
            self.forward_placeholders = {
              'inputs':
                  tf.placeholder(tf.float32, input_size + frame_shape),
              'input_action':
                  tf.placeholder(tf.float32, input_size + [self.num_acts]),
              'targets':
                  tf.placeholder(tf.float32, target_size + frame_shape),
              'target_action':
                  tf.placeholder(tf.float32, target_size + [self.num_acts]),
            }
            # Create model
            forward_model_cls = registry.model(FLAGS.model)
            forward_model = forward_model_cls(hparams, tf.estimator.ModeKeys.PREDICT)
            self.forward_prediction_ops, _ = forward_model(self.forward_placeholders)
            forward_saver = tf.train.Saver()
            forward_saver.restore(self.forward_sess,
                                self.sv2p_model_dir)
        print('LOADED SV2P!')


    def cem(self, args, forward_sess, forward_placeholders, forward_ops, curr, clusters, env, eps, planstep, verbose, trained_net,  sim_discriminator, transform=None):
        """Runs Visual MPC between two images."""
        horizon = forward_placeholders['targets'].shape[1]
        mu1 = np.array([0]*(self.num_acts * horizon))
        sd1 = np.array([0.2]*(self.num_acts * horizon))

        _iter = 0
        sample_size = self.cem_samples 
        resample_size = self.cem_samples // 5
        
        hz = horizon

        while np.max(sd1) > .001:
            if _iter == 0:
                acts1 = np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=[sample_size, hz, self.num_acts])
            else:
                acts1 = np.random.normal(mu1, sd1, (sample_size, self.num_acts *  hz))
            acts1 = acts1.reshape((sample_size, hz, self.num_acts))
            acts0 = acts1[:, 0:1, :]
            start = time.time()
            forward_feed = {
              forward_placeholders['inputs']:
                  np.repeat(np.expand_dims(np.expand_dims(curr, 0), 0),
                            sample_size, axis=0),
              forward_placeholders['input_action']:
                  acts0,
              forward_placeholders['targets']:
                  np.zeros(forward_placeholders['targets'].shape),
              forward_placeholders['target_action']:
                  acts1
            }
            forward_predictions = forward_sess.run(forward_ops, forward_feed)
            forward_predictions = forward_predictions.squeeze(-1)
            if not args.robot:
                forward_predictions = forward_predictions[:, :, :, 30:30+args.im_size, :]
            forward_predictions = forward_predictions.astype(np.uint8) / 255.
            preds = torch.FloatTensor(forward_predictions).permute(0, 1, 4, 2, 3).cuda() 
            transform = ComposeMix([
                [torchvision.transforms.Normalize(
                           mean=[0.485, 0.456, 0.406],  # default values for imagenet
                           std=[0.229, 0.224, 0.225]), "img"]
                 ])

            if transform is not None:                    
                preds = transform(preds, online=True)
            preds = [preds.permute(0, 2, 1, 3, 4)] 
            
            obs = []
            sub_len = 1
            rews = []
            demos = torch.tensor(clusters).cuda()
            demos = torch.cat(sub_len * [demos]).reshape(sub_len, -1, args.hidden_size)
            for p in range(args.sample_sz // 1): # in order to fit on the gpu
                ob = trained_net.encode([preds[0][p * sub_len: (p+1)*sub_len]])
                outputs = []
                for i in range(demos.shape[1]):
                    output = sim_discriminator.forward(ob, demos[:, i])
                    output = torch.nn.functional.softmax(output, dim=1)
                    outputs.append(output)
                outputs = torch.stack(outputs, axis=1)
                rew = np.mean(outputs.cpu().data.numpy(), axis=1) 
                rew = rew[:, 1] 
                rews.append(rew)
            rew = np.concatenate(rews)
            losses = rew
                    
            best_actions = np.array([x for _, x in sorted(
          zip(losses, acts1.tolist()), reverse=True)])
            best_costs = np.array([x for x, _ in sorted(
          zip(losses, acts1.tolist()), reverse=True)])

            start = time.time()
            """ Log top 5 and bottom 5 trajs """
            log_freq = 10 if not args.robot else 1
            if verbose > 0 and _iter == 0 and eps % log_freq == 0:
                for q in range(5):
                    head = self.savedir + 'rankings/{}/{}/'.format(eps, planstep)
                    if not os.path.exists(head):
                        os.makedirs(head)
                    if q == 0:
                        save_im(curr, head+'curr.jpg')
                    with imageio.get_writer('{}pred{}_{}.gif'.format(head, q, best_costs[q]), mode='I') as writer:
                        for p in range(horizon):
                            writer.append_data((forward_predictions[q, p, :, :, :] * 255.).astype('uint8'))
                for q in range(args.sample_sz-1, args.sample_sz-6, -1):
                    head = self.savedir + 'rankings/{}/{}/'.format(eps, planstep)
                    with imageio.get_writer('{}pred{}_{}.gif'.format(head, q, best_costs[q]), mode='I') as writer:
                        for p in range(horizon):
                            writer.append_data((forward_predictions[q, p, :, :, :] * 255.).astype('uint8'))
            
            best_actions = best_actions[:resample_size]
            best_actions1 = best_actions.reshape(resample_size, -1)
            if _iter < self.cem_iters:
                best_costs = best_costs[:resample_size]
                mu1 = np.mean(best_actions1, axis=0)
                sd1 = np.std(best_actions1, axis=0)
                _iter += 1
            else:
                break
          
        chosen = best_actions1[0]
        bestcost = best_costs[0]
        return chosen, bestcost


def main(argv=None):
    '''Initialize replay buffer, models, and environment.'''
    
    # Get args in argparser form
    args = get_args(FLAGS)
    args.sv2p_root += args.xml + '/'
    assert(torch.cuda.is_available())
    # Load in models and env
    print("----Load in models and env----")
    if args.similarity:
        cost = 'similarity'
    if not args.robot:
        env = Tabletop(
                        log_freq=args.env_log_freq, 
                        filepath=args.log_dir + '/env',
                        xml=args.xml,
                        verbose=args.verbose)
    else:
        from widowx_env.widowx import WidowX
        env = WidowX()
        args.action_dim = 4
        if not os.path.exists(args.log_dir + '/env'):
            os.mkdir(args.log_dir + '/env')
            
    sv2p = CEM(args, savedir=args.log_dir, phorizon=args.phorizon,
               cem_samples = args.sample_sz, cem_iters = args.cem_iters, cost=cost)
    
    print("----Done loading in models and env----")
    path = args.num_traj_per_epoch # num of trajs per episode
    hz = args.traj_length # traj_length
    full_low_dim = []
    
    ''' Initialize models '''
    model_dir = args.root + args.model_dir + 'model/'
    file_name = 'model3D_1'
    column_cnn_def = importlib.import_module("{}".format(file_name))
    
    trained_net = MultiColumn(args, args.num_tasks, column_cnn_def.Model, args.hidden_size)
    sim_discriminator = None
    
    # checkpoint path to a trained model
    checkpoint_path = os.path.join("pretrained/video_encoder/model_best.pth.tar")
    print("=> Checkpoint path --> {}".format(checkpoint_path))
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint")
        checkpoint = torch.load(checkpoint_path)
        checkpoint['state_dict'] = remove_module_from_checkpoint_state_dict(
                                              checkpoint['state_dict'])
        print("Loading in pretrained model")
        trained_net.load_state_dict(checkpoint['state_dict'], strict=False)
        if args.similarity:
            sim_discriminator = SimilarityDiscriminator(args).to(device)
            print("sim_discriminator", sim_discriminator)
            sim_discriminator.load_state_dict(torch.load(args.model_dir))
            sim_discriminator.eval()
    else:
        print(" !#! No checkpoint found at '{}'".format(
            checkpoint_path))

    print("=> loaded checkpoint '{}' (epoch {})"
          .format(checkpoint_path, checkpoint['epoch']))
    trained_net.eval()
    trained_net.cuda()
    
    transform = ComposeMix([
        [Scale(args.im_size), "img"],
        [torchvision.transforms.ToPILImage(), "img"],
        [torchvision.transforms.CenterCrop(args.im_size), "img"],
        [torchvision.transforms.ToTensor(), "img"],
        [torchvision.transforms.Normalize(
                   mean=[0.485, 0.456, 0.406],  # default values for imagenet
                   std=[0.229, 0.224, 0.225]), "img"]
         ])
    
    if not os.path.exists(args.log_dir + 'rankings/'):
        os.mkdir(args.log_dir + 'rankings/')
    
    # Get clusters
    clusters = None
    if not args.random and args.similarity:
        def get_cluster(demo, clusters, vids, save_demo=True):
            one_shot_demo = args.demo_path + str(vids[demo]) + '.webm'
            reader = av.open(one_shot_demo)
            imgs = []
            imgs = [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]
            # Downsample imgs
            downsample = max(1, len(imgs) // 30)
            imgs = imgs[1::downsample][:40]
            if save_demo:
                orig_imgs = np.array(imgs).copy()
                with imageio.get_writer(args.log_dir + '/demo' + str(vids[demo]) + '.gif', mode='I') as writer:
                    for k, frame in enumerate(orig_imgs):
                        img = Scale(args.im_size)(frame).astype('uint8')
                        writer.append_data(img)
            if transform:
                imgs = transform(imgs)
            imgs = torch.stack(imgs).permute(1, 0, 2, 3).unsqueeze(0) # want 1, 3, traj_length, 84, 84
            input_var = [imgs.to(device)]
            features = trained_net.encode(input_var)
            features = features.cpu().data.numpy()
            clusters.append(features)

        clusters = []
        vids = [num for num in range(args.num_demos)]
        for d in range(args.num_demos):
            get_cluster(d, clusters, vids)
        clusters = np.concatenate(clusters, axis=0)            

    env.max_path_length = args.traj_length * args.num_traj_per_epoch
    report_losses = {}
    report_losses['dynamics_loss'] = []
    
    if not args.robot:
        env.initialize()
    total_good = 0
    results = []
    save_freq = 10 if not args.robot else 1
    for eps in range(args.num_epochs):
        eps_low_dim = []
        start = time.time()

        obs, env_info = env.reset_model()
        init_im = obs * 255 
        if eps == 0 and args.verbose:
            save_im(init_im, '{}/init.png'.format(args.log_dir))

        if args.robot:
            for _ in range(4):
                obs, reward, terminal, action, succ = env.step([0, 0, 0, 0])
                if eps == 0 and args.verbose:
                    save_im(obs*255, '{}/init.png'.format(args.log_dir))
            
        step = 0
        if not args.robot:
            low_dim_state = get_obs(args, env_info)
            very_start = low_dim_state
            eps_low_dim.append(low_dim_state)
        while step < path: # each episode is 3 x 20-step trajectories
            if step == 0:
                obs = obs * 255
            step_time = time.time()
            if args.random:
                chosen = np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=[hz, args.action_dim])
                chosen = chosen.reshape(hz*args.action_dim, -1).squeeze()
                bestcost = 0
            else:
                chosen, bestcost = sv2p.cem(args, sv2p.forward_sess, sv2p.forward_placeholders, 
                                        sv2p.forward_prediction_ops, obs, clusters, 
                                        env, eps, step, args.verbose, trained_net, sim_discriminator, transform=transform)
            for h in range(hz): 
                if args.robot:
                    obs, reward, terminal, action, succ = env.step(chosen[args.action_dim*h:(args.action_dim)*(h+1)])
                    # if got error
                    if not succ:
                        print("Got " + str(eps) + " epochs")
                        assert(False)
                else:
                    obs, reward, terminal, env_info = env.step(chosen[args.action_dim*h:(args.action_dim)*(h+1)])
                obs = obs * 255
                if args.verbose and eps % save_freq == 0:
                    save_im(obs, '{}step{}.png'.format(args.log_dir, step * hz + h))
                    
                if not args.robot:
                    low_dim_state = get_obs(args, env_info)
                    eps_low_dim.append(low_dim_state)
            step += 1
            
        if args.verbose and eps % save_freq == 0:
            total_steps = args.traj_length * args.num_traj_per_epoch
            if args.robot:
                for p in range(4):
                    obs, reward, terminal, action, succ = env.step([0, 0, 0, 0])
                    save_im(obs*255, '{}step{}.png'.format(args.log_dir, args.traj_length * args.num_traj_per_epoch +p))
                total_steps += 4
            with imageio.get_writer('{}{}.gif'.format(args.log_dir, eps), mode='I', fps=8) as writer:
                for step in range(total_steps):
                    img_path = '{}step{}.png'.format(args.log_dir, step)
                    writer.append_data(imageio.imread(img_path))

        if not args.robot:
            full_low_dim.append(np.array(eps_low_dim))
        else:
            criteria = input(f"Was task {args.task_num} completed?")
            if int(criteria) == 1:
                total_good += 1
                results.append(1)
            else:
                results.append(0)
        end = time.time()
        print("Time for 1 trial", end - start)
        print("-----------------EPS {} ENDED--------------------".format(eps))
        if args.robot:
            time.sleep(10)
    if not args.robot:
        import pickle
        pickle.dump(np.array(full_low_dim), open(args.log_dir + 'full_states.p', 'wb'))
    else:
        print("Total successes", total_good)
        print("Results", results)
        np.savetxt(args.log_dir + 'results.txt', np.array(results), fmt='%d')
            

if __name__ == "__main__":
    app.run(main)
        
    