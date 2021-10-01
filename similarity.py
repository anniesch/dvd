import os
import sys
import time
import signal
import importlib
import threading

import torch
import torch.nn as nn
import numpy as np

from utils import *
from callbacks import (PlotLearning, AverageMeter)
from transforms_video import *
import cv2
import imageio
import pickle
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

def train_similarity(args, train_loader, model, sim_discriminator, loss_class, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    class_losses = AverageMeter()
    false_pos_meter = AverageMeter()
    false_neg_meter = AverageMeter()
    torch.autograd.set_detect_anomaly(True)

    # switch to train mode
    model.train()
    sim_discriminator.train()

    end = time.time()
    # Want random length trajectories if video enc
    if args.traj_length == 0:
        rand_length = np.random.randint(20, 40)
        train_loader.dataset.traj_length = rand_length
        print("training rand length:", train_loader.dataset.traj_length)
    
    len_dataloader =len(train_loader) 
    data_source_iter = iter(train_loader)
    i = 0
    while i < len_dataloader:
        # training model using source data (Human data here, has labeled tasks)
        data_source = data_source_iter.next()
        pos_data, anchor_data, neg_data = data_source
        # measure data loading time
        data_time.update(time.time() - end)
        
        pos_data = [pos_data.to(device)]
        anchor_data = [anchor_data.to(device)]
        neg_data = [neg_data.to(device)]

        model.zero_grad()
        sim_discriminator.zero_grad()
        
        # Encode videos
        pos_enc = model.encode(pos_data)
        anchor_enc = model.encode(anchor_data)
        neg_enc = model.encode(neg_data)
        
        # Calculate loss
        pos_anchor = sim_discriminator.forward(pos_enc, anchor_enc)
        neg_anchor = sim_discriminator.forward(anchor_enc, neg_enc)
        pos_anchor_label = torch.ones(args.batch_size).long().cuda()
        neg_anchor_label = torch.zeros(args.batch_size).long().cuda()
        class_out = torch.cat((pos_anchor, neg_anchor))  
        sim_labels = torch.cat((pos_anchor_label, neg_anchor_label))
        class_loss = loss_class(class_out, sim_labels)
                       
        enc = torch.cat((pos_enc, anchor_enc, neg_enc))
        loss = class_loss

        # measure accuracy and record loss
        prec1 = float(((class_out[:, 0] < class_out[:, 1]) == sim_labels[:]).sum()) / class_out.shape[0]
        top1.update(prec1, 1) #class_out.size(0)
        losses.update(loss.item(), 1)
        class_losses.update(class_loss.item(), 1)
        false_pos = float(((class_out[:, 0] < class_out[:, 1]) & (sim_labels[:] == 0)).sum()) / float((sim_labels[:] == 0).sum())
        false_neg = float(((class_out[:, 0] > class_out[:, 1]) & (sim_labels[:] == 1)).sum()) / float((sim_labels[:] == 1).sum())
        false_pos_meter.update(false_pos, 1)
        false_neg_meter.update(false_neg, 1)

        # compute gradient and do SGD step for task classifier
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Acc {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Classloss {class_losses.val:.3f} ({class_losses.avg:.3f})\t'.format(
                      epoch, i, len_dataloader, batch_time=batch_time,
                      data_time=data_time, top1=top1, loss=losses, class_losses=class_losses))
        i += 1
    return losses.avg, top1.avg, class_losses.avg, false_pos_meter.avg, false_neg_meter.avg


def validate_similarity(args, val_loader, model, sim_discriminator, loss_class, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    false_pos_meter = AverageMeter()
    false_neg_meter = AverageMeter()

    # switch to evaluate mode
    model.eval()
    sim_discriminator.eval()

    # Want random length trajectories
    if args.traj_length == 0:
        rand_length = np.random.randint(20, 40)
        val_loader.dataset.traj_length = rand_length
    
    end = time.time()
    with torch.no_grad():
        data_source_iter = iter(val_loader)
        i = 0
        while i < len(val_loader):
            data_source = data_source_iter.next()
            pos_data, anchor_data, neg_data = data_source

            pos_data = [pos_data.to(device)]
            anchor_data = [anchor_data.to(device)]
            neg_data = [neg_data.to(device)]

            # Encode videos
            pos_enc = model.encode(pos_data)
            anchor_enc = model.encode(anchor_data)
            neg_enc = model.encode(neg_data)
        
            # Calculate loss
            pos_anchor = sim_discriminator.forward(pos_enc, anchor_enc)
            neg_anchor = sim_discriminator.forward(anchor_enc, neg_enc)
            pos_anchor_label = torch.ones([args.batch_size])
            neg_anchor_label = torch.zeros([args.batch_size])
            pos_neg_label = torch.zeros([args.batch_size])
            class_out = torch.cat((pos_anchor, neg_anchor)) 
            sim_labels = torch.cat((pos_anchor_label, pos_neg_label)).long().cuda()
            class_loss = loss_class(class_out, sim_labels)
            
            enc = torch.cat((pos_enc, anchor_enc, neg_enc))
            
            prec1 = float(((class_out[:, 0] < class_out[:, 1]) == sim_labels[:]).sum()) / class_out.shape[0]
            false_pos = float(((class_out[:, 0] < class_out[:, 1]) & (sim_labels[:] == 0)).sum()) / float((sim_labels[:] == 0).sum())
            false_neg = float(((class_out[:, 0] > class_out[:, 1]) & (sim_labels[:] == 1)).sum()) / float((sim_labels[:] == 1).sum())
            false_pos_meter.update(false_pos, 1)
            false_neg_meter.update(false_neg, 1)
            top1.update(prec1, 1)
            losses.update(class_loss.item(), 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          top1=top1))
            i += 1
        
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return losses.avg, top1.avg, false_pos_meter.avg, false_neg_meter.avg