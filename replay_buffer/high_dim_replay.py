import numpy as np
import random
import h5py
import os

class ImageBuffer():
    def __init__(
            self,
            args,
            trajectory_length,
            action_dim,
            savedir,
            memory_size,
            state_dim=120, #84, #240, #64,
            ):
        super().__init__()
        # divide by trajectory length so that each buffer has max_replay_buffer_size total states
        self.memory_size = memory_size
        self.trajectory_length = trajectory_length
        self.state_dim = state_dim
        self.state_dim_x = int(state_dim * 1.5)
        self.action_dim = action_dim
        self.filepath = savedir + '/img_memory/'
        if not os.path.exists(self.filepath):
            os.mkdir(self.filepath)
        
        self._top = 0 # Keep track of place in buffer
        self._size = 0 # Keep track of size of memory
        self._top_mem = 0 # Keep track of place in memory
        self.iter = 0 # Keep track of saved buffer num
        
        self.action_memory = np.zeros((self.memory_size, self.trajectory_length, self.action_dim), dtype=float)
        self.next_state_memory = np.zeros((self.memory_size, self.trajectory_length, 3, self.state_dim, self.state_dim_x), dtype=float)
        self.state_memory = np.zeros((self.memory_size, self.trajectory_length, 3, self.state_dim, self.state_dim_x), dtype=float)

        self.f = []
        self.f_ptr = h5py.File(self.filepath + str(self.iter) + 'mem.hdf5', 'w') #, HDF5_USE_FILE_LOCKING='FALSE')
        self.sim_data = self.f_ptr.create_group('sim')
        # state includes: gripper xyz, (block xyz, dx dy dz) x 3 blocks, gleft, gright, z joint: 24
        self.sim_data.create_dataset('states', (self.memory_size, self.trajectory_length, 3, self.state_dim, self.state_dim_x), dtype='f')
        self.sim_data.create_dataset('next_states', (self.memory_size, self.trajectory_length, 3, self.state_dim, self.state_dim_x), dtype='f')
        self.sim_data.create_dataset('actions', (self.memory_size, self.trajectory_length, self.action_dim), dtype='f')
        self.f.append(self.f_ptr)


    def add_sample(self, states, next_states, actions):
        self.action_memory[self._top_mem,:,:] = np.array(actions)
        self.next_state_memory[self._top_mem,:,:,:,:] = np.array(next_states)
        self.state_memory[self._top_mem,:,:,:,:] = np.array(states)
        if self._size < self.memory_size:
            self._size += 1

        with h5py.File(self.filepath + str(self.iter) + 'mem.hdf5', 'a') as self.f_ptr:
            self.f_ptr['sim']['states'][self._top] = states
            self.f_ptr['sim']['actions'][self._top] = actions
            self.f_ptr['sim']['next_states'][self._top] = next_states
        self._top += 1
        self._top_mem += 1

        if self._top >= self.memory_size:
            self.f_ptr.close()
            self.iter += 1
            print("starting new img memory")
            self.f_ptr = h5py.File(self.filepath + str(self.iter) + 'mem.hdf5', 'w') #, HDF5_USE_FILE_LOCKING='FALSE')
            self.sim_data = self.f_ptr.create_group('sim')
            self.sim_data.create_dataset('states', (self.memory_size, self.trajectory_length, 3, self.state_dim, self.state_dim_x), dtype='f')
            self.sim_data.create_dataset('next_states', (self.memory_size, self.trajectory_length, 3, self.state_dim, self.state_dim_x), dtype='f')
            self.sim_data.create_dataset('actions', (self.memory_size, self.trajectory_length, self.action_dim), dtype='f')
            self.f.append(self.f_ptr)
            self._top = 0

        if self._top_mem >= self.memory_size - 1:
            self._top_mem = 0

    """
    Draw batch of states and next states: K steps x 3 x 48 x 48
    """
    def draw_samples(self,batch_size,length):
        indices = np.random.choice(self._size, batch_size, replace=True)
        states = np.take(self.state_memory, indices, axis=0, out=None)
        next_states = np.take(self.next_state_memory, indices, axis=0, out=None)
        actions = np.take(self.action_memory, indices, axis=0, out=None)
        seqs = []
        acts = []
        nexts = []
        for b in range(batch_size):
            start = np.random.randint(0, states.shape[1] - length)
            select_obs = states[b,start:start+length,:]
            select_next = next_states[b,start:start+length,:]
            select_act = actions[b,start:start+length,:]
            seqs.append(select_obs)
            nexts.append(select_next)
            acts.append(select_act)
        seqs = np.array(seqs)
        nexts = np.array(nexts)
        acts = np.array(acts)
        return seqs, nexts, acts, True
