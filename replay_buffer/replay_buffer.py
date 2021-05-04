import numpy as np
import random
import h5py
import os

""" 
Only use low dim replay buffer for counting target interaction.
Use high_dim_replay for model training
"""
class ReplayBuffer():
    def __init__(
            self,
            max_replay_buffer_size,
            trajectory_length,
            state_dim,
            action_dim,
            savedir,
            ):
        super().__init__()
        self.max_replay_buffer_size = max_replay_buffer_size
        self.trajectory_length = trajectory_length
        self.filepath = savedir + '/memory/'
        if not os.path.exists(self.filepath):
            os.mkdir(self.filepath)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._top = 0 # Keep track of place in buffer
        self.iter = 0 # keep track of saved buffer num
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        f_ptr = h5py.File(self.filepath + str(self.iter) + '.hdf5', 'w') #, HDF5_USE_FILE_LOCKING='FALSE')
        self.sim_data = f_ptr.create_group('sim')
        # state includes: gripper xyz, (block xyz, dx dy dz) x 3 blocks, gleft, gright, z joint: 24
        self.sim_data.create_dataset('states', (self.max_replay_buffer_size, self.trajectory_length, self.state_dim), dtype='f')
        self.sim_data.create_dataset('next_states', (self.max_replay_buffer_size, self.trajectory_length, self.state_dim), dtype='f')
        self.sim_data.create_dataset('actions', (self.max_replay_buffer_size, self.trajectory_length, self.action_dim), dtype='f')

    def add_sample(self, states, next_states, actions):
        with h5py.File(self.filepath + str(self.iter) + '.hdf5', 'a') as f_ptr:
            f_ptr['sim']['states'][self._top, :, :] = states
            f_ptr['sim']['actions'][self._top, :, :] = actions
            f_ptr['sim']['next_states'][self._top, :, :] = next_states
        self._top += 1
        if self._top >= self.max_replay_buffer_size:
            f_ptr.close()
            self.iter += 1
            f_ptr = h5py.File(self.filepath + str(self.iter) + '.hdf5', 'w')
            self.sim_data = f_ptr.create_group('sim')
            self.sim_data.create_dataset('states', (self.max_replay_buffer_size, self.trajectory_length, self.state_dim), dtype='f')
            self.sim_data.create_dataset('actions', (self.max_replay_buffer_size, self.trajectory_length, self.action_dim), dtype='f')
            self.sim_data.create_dataset('next_states', (self.max_replay_buffer_size, self.trajectory_length, self.state_dim), dtype='f')
            self._top = 0 
