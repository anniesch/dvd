import os
import sys
import signal
import importlib

import numpy as np
import torch
import torch.nn as nn

from utils import load_args, setup_cuda_devices
from callbacks import (PlotLearning, AverageMeter)
from multi_column import MultiColumn, SimilarityDiscriminator
import torchvision
from transforms_video import ComposeMix, RandomCropVideo, RandomRotationVideo, Scale
from data_loader_av import VideoFolder
import cv2
import imageio
import pickle
import json
from PIL import Image

from similarity import train_similarity, validate_similarity 


def main():
    # load args
    args = load_args()

    # set column model
    cnn_def = importlib.import_module("{}".format('model3D_1'))

    # setup device - CPU or GPU
    device, device_ids = setup_cuda_devices(args)
    print(" > Using device: {}".format(device.type))
    print(" > Active GPU ids: {}".format(device_ids))

    best_loss = float('Inf')

    args.human_tasks = [int(i) for i in args.human_tasks]
    args.robot_tasks = [int(i) for i in args.robot_tasks]
    args.num_tasks = len(args.human_tasks)
    if args.just_robot:
        args.num_tasks = len(args.robot_tasks)
        args.human_tasks = args.robot_tasks
    # set run output folder
    save_dir = args.log_dir + 'tasks' + str(args.num_tasks) + '_seed' + str(args.seed) + '_lr' + str(args.lr)
    if args.traj_length != 0:
        save_dir +='_traj' + str(args.traj_length)
    if args.similarity:
        save_dir += '_sim'
    if args.pretrained:
        save_dir += '_pre'
    save_dir += '_hum'
    for num in args.human_tasks:
        save_dir += str(num) 
    if args.add_demos:
        save_dir += '_dem' + str(args.add_demos) + '_rob'
        for num in args.robot_tasks:
            save_dir += str(num)
    if args.im_size != 120:
        save_dir += '_im' + str(args.im_size)
    if args.just_robot:
        save_dir += '_justrobot'
    print(" > Output folder for this run -- {}".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'plots'))
        os.makedirs(os.path.join(save_dir, 'model'))
    with open(save_dir + '/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    args.log_dir = save_dir

    # create model
    print(" > Creating model ... !")
    model = MultiColumn(args, args.num_tasks, cnn_def.Model,
                        int(args.hidden_size))

    if args.resume or args.pretrained: # optionally resume from a checkpoint
        if args.pretrained:
            checkpoint_path = os.path.join(args.pretrained_dir, "model_best.pth.tar")
        else:
            checkpoint_path = os.path.join(args.log_dir, 'model',
                                   str(args.resume) + 'checkpoint.pth.tar')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            if args.pretrained:
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

                print("Loading in pretrained model")
                checkpoint['state_dict'] = remove_module_from_checkpoint_state_dict(
                                          checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(" > Loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print(" !#! No checkpoint found at '{}'".format(
                checkpoint_path))
            assert(False)
    model = model.to(device)
        
    if args.similarity:
        sim_discriminator = SimilarityDiscriminator(args).to(device)
    if args.pretrained:
        for p in model.parameters():
            p.requires_grad = False
            
    print("Trainable params in encoder:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # define augmentation pipeline
    upscale_size_train = int(args.im_size * 1.4)
    upscale_size_eval = int(args.im_size * 1)

    # Random crop videos during training
    transform_train_pre = ComposeMix([
            [RandomRotationVideo(15), "vid"],
            [Scale(upscale_size_train), "img"],
            [RandomCropVideo(args.im_size), "vid"],
             ])

    # Center crop videos during evaluation
    transform_eval_pre = ComposeMix([
            [Scale(upscale_size_eval), "img"],
            [torchvision.transforms.ToPILImage(), "img"],
            [torchvision.transforms.CenterCrop(args.im_size), "img"],
             ])

    # Transforms common to train and eval sets and applied after "pre" transforms
    transform_post = ComposeMix([
            [torchvision.transforms.ToTensor(), "img"],
            [torchvision.transforms.Normalize(
                       mean=[0.485, 0.456, 0.406],  # default values for imagenet
                       std=[0.229, 0.224, 0.225]), "img"]
             ])
    
    # Transform for robot demos
    robot_demo_transform = ComposeMix([
        [RandomRotationVideo(15), "vid"],
        [torchvision.transforms.ToPILImage(), "img"],
        [torchvision.transforms.CenterCrop(args.im_size), "img"],
        [torchvision.transforms.ToTensor(), "img"],
        [torchvision.transforms.Normalize(
                   mean=[0.485, 0.456, 0.406],  # default values for imagenet
                   std=[0.229, 0.224, 0.225]), "img"]
         ])        

    train_data = VideoFolder(args,
                             root=args.human_data_dir,
                             json_file_input=args.json_data_train,
                             json_file_labels=args.json_file_labels,
                             clip_size=args.traj_length,
                             nclips=1,
                             step_size=1,
                             num_tasks=args.num_tasks,
                             is_val=False,
                             transform_pre=transform_train_pre,
                             transform_post=transform_post,
                             robot_demo_transform=robot_demo_transform,
                             )

    print(" > Using {} processes for data loader.".format(2))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
        drop_last=True)

    val_data = VideoFolder(args, 
                           root=args.human_data_dir,
                           json_file_input=args.json_data_val,
                           json_file_labels=args.json_file_labels,
                           clip_size=args.traj_length,
                           nclips=1,
                           step_size=1,
                           num_tasks=args.num_tasks,
                           is_val=True,
                           transform_pre=transform_eval_pre,
                           transform_post=transform_post,
                           robot_demo_transform=robot_demo_transform,
                           )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
        drop_last=True)

    print(" > Number of dataset classes : {}".format(len(train_data.classes_dict.keys())//2))
    assert len(train_data.classes_dict.keys())//2 == args.num_tasks

    # define loss function (criterion)
    loss_class = nn.CrossEntropyLoss().to(device)

    # define optimizer
    lr = args.lr
    last_lr = 1e-05
    params = list(model.parameters())
    if args.similarity:
        params += list(sim_discriminator.parameters())
        print("Number of discriminator params", sum(p.numel() for p in sim_discriminator.parameters() if p.requires_grad))
        optimizer = torch.optim.SGD(params, lr,
                                 momentum=0.9,
                                 weight_decay=0.00001)

    # set callbacks
    plotter = PlotLearning(args, os.path.join(
        args.log_dir, "plots"), args.num_tasks)
    lr_decayer = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, 'min', factor=0.5, patience=5, verbose=True)
    val_loss = float('Inf')

    print(" > Training is getting started...")
    print(" > Training takes {} epochs.".format(args.num_epochs))
    start_epoch = args.resume if args.resume > 0 else 0
    
    report_losses = {}
    report_losses['train_acc'] = []
    report_losses['val_acc'] = []
    report_losses['val_loss'] = []
    report_losses['train_loss'] = []
    report_losses['false_pos'] = []
    report_losses['false_neg'] = []
    report_losses['false_pos_train'] = []
    report_losses['false_neg_train'] = []

    for epoch in range(start_epoch, args.num_epochs):

        lrs = [params['lr'] for params in optimizer.param_groups]
        print(" > Current LR(s) -- {}".format(lrs))
        if np.max(lr) < last_lr and last_lr > 0:
            print(" > Training is DONE by learning rate {}".format(last_lr))
            sys.exit(1)

        if args.similarity:
            train_loss, train_top1, class_loss, false_pos_train, false_neg_train = train_similarity(args, 
            train_loader, model, sim_discriminator, loss_class, optimizer, epoch)
        
        # evaluate on validation set
        if epoch % args.log_freq == 0:
            print("Evaluating on epoch", epoch)
            if args.similarity:
                val_loss, val_top1, false_pos, false_neg = validate_similarity(args, val_loader, model, sim_discriminator, loss_class, epoch)
                
            # set learning rate
            lr_decayer.step(val_loss)

            # plot learning
            plotter_dict = {}
            plotter_dict['loss'] = train_loss
            plotter_dict['val_loss'] = 0 
            plotter_dict['class_loss'] = class_loss
            plotter_dict['val_acc'] = val_top1 
            plotter_dict['learning_rate'] = lr
            plotter_dict['false_pos_train'] = false_pos_train
            plotter_dict['false_neg_train'] = false_neg_train
            plotter_dict['false_pos'] = false_pos
            plotter_dict['false_neg'] = false_neg
            plotter_dict['val_loss'] = val_loss
            plotter_dict['acc'] = train_top1
            plotter_dict['val_acc'] = val_top1
            
            plotter.plot(plotter_dict)
            
            report_losses['val_acc'].append(val_top1)
            report_losses['train_acc'].append(train_top1)
            report_losses['val_loss'].append(val_loss)
            np.savetxt(args.log_dir + '/val_acc.txt', np.array(report_losses['val_acc']), fmt='%f')
            np.savetxt(args.log_dir + '/train_acc.txt', np.array(report_losses['train_acc']), fmt='%f')
            np.savetxt(args.log_dir + '/val_loss.txt', np.array(report_losses['val_loss']), fmt='%f')
            if args.similarity:
                report_losses['false_pos'].append(false_pos)
                report_losses['false_neg'].append(false_neg)
                report_losses['false_pos_train'].append(false_pos_train)
                report_losses['false_neg_train'].append(false_neg_train)
                np.savetxt(args.log_dir + '/false_pos.txt', np.array(report_losses['false_pos']), fmt='%f')
                np.savetxt(args.log_dir + '/false_neg.txt', np.array(report_losses['false_neg']), fmt='%f')
                np.savetxt(args.log_dir + '/false_pos_train.txt', np.array(report_losses['false_pos_train']), fmt='%f')
                np.savetxt(args.log_dir + '/false_neg_train.txt', np.array(report_losses['false_neg_train']), fmt='%f')
                
            print(" > Validation accuracy after epoch {} = {}".format(epoch, val_top1))

            # remember best loss and save the checkpoint
            freq = 10 if args.similarity else 5
            if (epoch + 1) % freq == 0:
                is_best = val_loss < best_loss
                best_loss = min(val_loss, best_loss)

                if args.similarity: # need to save sim discriminator
                    save_path = os.path.join(args.log_dir, 'model', str(epoch+1) + 'sim_discriminator.pth.tar')
                    torch.save(sim_discriminator.state_dict(), save_path)
            

if __name__ == '__main__':
    main()
