# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import video_utils
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
import h5py
import tensorflow as tf

NUMEP = 500 # Each im buffer has 500 eps: 5 trajectories of 10 steps each = 2500 trajectories
EPLEN = 60 # Needs to be 50, should loop through 5 10-step trajs at a time 

@registry.register_problem
class HumanUpdated(video_utils.VideoProblem):

    @property
    def num_channels(self):
        return 3

    @property
    def frame_height(self):
        return 120

    @property
    def frame_width(self):
        return 180

    @property
    def is_generate_per_split(self):
        return True

    # num_hdf * 25000 (num of images per image memory hdf = NUMEP * EPLEN)
    @property
    def total_number_of_frames(self):
        return 60*20000 #10k random center + 5k random left & 5k random right

    # Not sure if this is correct? We don't have videos
    def max_frames_per_video(self, hparams):
        return 60

    @property
    def random_skip(self):
        return False

    @property
    def only_keep_videos_from_0th_frame(self):
        return False

    @property
    def use_not_breaking_batching(self):
        return True

    @property
    def extra_reading_spec(self):
        """Additional data fields to store on disk and their decoders."""
        data_fields = {
            "frame_number": tf.FixedLenFeature([1], tf.int64),
            "action":tf.FixedLenFeature([5], tf.float32),
        }
        decoders = {
            "frame_number": tf.contrib.slim.tfexample_decoder.Tensor(
                tensor_key="frame_number"),
            "action": tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="action"),
        }
        return data_fields, decoders

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.modality = {"inputs": modalities.ModalityType.VIDEO,
                      "action":modalities.ModalityType.REAL_L2_LOSS,
                      "targets": modalities.ModalityType.VIDEO}
        p.vocab_size = {"inputs": 256, 
                        "action": 5,
                        "targets": 256}

    def parse_frames(self, f, dataset_split):
        ims = f['sim']['states'][:]
        next_ims = f['sim']['next_states'][:]
        acts = f['sim']['actions'][:]
        
        ims = np.transpose(ims, (0, 1, 3, 4, 2)) # Should be (500, 60, 120, 180, 3)

        if dataset_split == problem.DatasetSplit.TRAIN:
            start_ep, end_ep = 0, int(NUMEP * 0.8) # 400 eps 
        else:
            start_ep, end_ep = int(NUMEP * 0.8), NUMEP # 100
            
        for ep in range(start_ep, end_ep): # goes from 0 to 399, each 60 step eps
            for step in range(EPLEN): # Go through the 60 steps of the episode
                frame = ims[ep, step] * 255.
                action = acts[ep, step]
                yield step, frame, action

                    
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        
        for j in range(20): # Load in random data
            path = 'PATH/img_memory/{}mem.hdf5'.format(j)
            f = h5py.File(path, "r")
            for frame_number, frame, action in self.parse_frames(f, dataset_split): # frame number needs to be 0, ..., 59
                yield {
                    "frame_number": [frame_number],
                    "frame": frame,
                    "action": action.tolist(),
                }
                    