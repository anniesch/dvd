import sys
import time
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
from torch.optim.optimizer import Optimizer


###############################################################################
# TRAINING CALLBACKS
###############################################################################

class PlotLearning(object):
    def __init__(self, args, save_path, num_classes):
        self.accuracy = []
        self.val_accuracy = []
        self.losses = []
        self.val_losses = []
        self.learning_rates = []
        self.similarity = args.similarity
        if self.similarity:
            self.false_pos_train = []
            self.false_neg_train = []
            self.false_pos_val = []
            self.false_neg_val = []
        self.save_path_loss = os.path.join(save_path, 'loss_plot.png')
        self.save_path_accu = os.path.join(save_path, 'accu_plot.png')

    def plot(self, logs):
        self.accuracy.append(logs.get('acc'))
        self.val_accuracy.append(logs.get('val_acc'))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.learning_rates.append(logs.get('learning_rate'))
        
        if self.similarity:
            self.false_pos_train.append(logs.get('false_pos_train'))
            self.false_neg_train.append(logs.get('false_neg_train'))
            self.false_pos_val.append(logs.get('false_pos'))
            self.false_neg_val.append(logs.get('false_neg'))

        best_val_acc = max(self.val_accuracy)
        best_train_acc = max(self.accuracy)
        best_val_epoch = self.val_accuracy.index(best_val_acc)
        best_train_epoch = self.accuracy.index(best_train_acc)

        plt.figure(1)
        plt.gca().cla()
        plt.ylim(0, 1)
        plt.plot(self.accuracy, label='train')
        plt.plot(self.val_accuracy, label='valid')
        
        if self.similarity:
            plt.plot(self.false_pos_train, label='false_pos_train')
            plt.plot(self.false_neg_train, label='false_neg_train')
            plt.plot(self.false_pos_val, label='false_pos_val')
            plt.plot(self.false_neg_val, label='false_neg_val')
        plt.title("best_val@{0:}-{1:.2f}, best_train@{2:}-{3:.2f}".format(
            best_val_epoch, best_val_acc, best_train_epoch, best_train_acc))
        plt.legend()
        plt.savefig(self.save_path_accu)

        best_val_loss = min(self.val_losses)
        best_train_loss = min(self.losses)
        best_val_epoch = self.val_losses.index(best_val_loss)
        best_train_epoch = self.losses.index(best_train_loss)

        plt.figure(2)
        plt.gca().cla()
        plt.plot(self.losses, label='train')
        plt.plot(self.val_losses, label='valid')
        plt.title("best_val@{0:}-{1:.2f}, best_train@{2:}-{3:.2f}".format(
            best_val_epoch, best_val_loss, best_train_epoch, best_train_loss))
        plt.legend()
        plt.savefig(self.save_path_loss)


# Taken from PyTorch's examples.imagenet.main
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        