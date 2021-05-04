import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os


def _count(buffer_id, args, close_drawer, faucet_right, faucet_left, cup_forward, target_dists, avg_drawer_last, avg_faucet_last, target_avgs, sv2p=False):
    threshold = args.threshold
    if not sv2p:
        f = h5py.File(buffer_id + '.hdf5', 'r')
        qpos_init = f['sim']['states'][6, 0, :]
        print("len", len(qpos_init), qpos_init)
        qpos = f['sim']['states'][:, :, :]
        buffer_size = qpos.shape[0] // args.num_traj_per_epoch
    else:
        with open(buffer_id + '/full_states.p', 'rb') as f: # Try different thresholds for final dists
            qpos = pickle.load(f)
        print("states", qpos.shape)
        qpos_init = qpos[6, 0, :]
        buffer_size = args.n_trials
    if 'env4' in buffer_id: 
        target = [0.17, .65, 0] 
    elif 'env3' in buffer_id: 
        target = [-0.05, .7, 0] 
    else:
        target = [0, .7, 0]
    right_to_left = qpos[:,:,3:4] - qpos_init[3:4]
    left_to_right = -qpos[:,:,3:4] + qpos_init[3:4]
    right_to_left = right_to_left.reshape(buffer_size, -1) # 100 is size of low-dim buffer 
    left_to_right = left_to_right.reshape(buffer_size, -1)
    right_to_left_n = sum(np.max(right_to_left[:args.n_trials], -1) > args.threshold)
    left_to_right_n = sum(np.max(left_to_right[:args.n_trials], -1) > args.threshold)

    forward = qpos[:,:,4:5] - qpos_init[4:5]
    forward = np.max(forward.reshape(buffer_size, -1), -1)
    push_forward = sum((abs(np.max(right_to_left[:args.n_trials], -1)) < 0.05)  & (forward[:args.n_trials] > args.threshold)) # 0.05
    
    target_dist = np.sqrt(np.sum((qpos[:,:,3:6] - target)**2, axis=2)).reshape(buffer_size, -1)
    target_dist_thres = sum(np.min(target_dist[:args.n_trials], -1) < 0.075) 
    
    drawer_move = np.abs(qpos[:,:,10] - qpos_init[10]).reshape(buffer_size, -1)
    coffee_move = np.abs(qpos[:,:,11] - qpos_init[11]).reshape(buffer_size, -1)
    faucet_move = (qpos[:,:,12] - qpos_init[12]).reshape(buffer_size, -1)
    drawer_closed = np.abs(qpos[:,:,10]).reshape(buffer_size, -1)
    drawer_closed_n = sum(drawer_closed[:args.n_trials, -1] < 0.05) 
    drawer_move_n = sum(np.max(drawer_move[:args.n_trials], -1) > 0.06) 
    coffee_move_n = sum(np.max(coffee_move[:args.n_trials], -1) > 0.01)
    faucet_left_n = sum(np.max(faucet_move[:args.n_trials], -1) > 0.01) 
    faucet_right_n = sum(np.min(faucet_move[:args.n_trials], -1) < -0.01)
    
    close_drawer.append(drawer_closed_n)
    faucet_right.append(faucet_right_n)
    faucet_left.append(faucet_left_n)
    cup_forward.append(push_forward)
    target_dists.append(target_dist_thres)
    target_avg = np.mean(np.min(target_dist[:args.n_trials], -1))
    target_avgs.append(target_avg)
    drawer_last = np.mean(drawer_closed[:args.n_trials, -1])
    drawer_last = np.mean(np.min(drawer_closed[:args.n_trials], -1))
    avg_drawer_last.append(drawer_last)
    faucet_last = np.abs(qpos[:,:,12]).reshape(buffer_size, -1)
    faucet_last = np.mean(faucet_last[:args.n_trials, -1])
    avg_faucet_last.append(faucet_last)

    print("Left to right (task 93): ", left_to_right_n, " | Right to left (task 94): ", right_to_left_n, " | Forward: ", push_forward, " | Target pos: ", target_dist_thres, " | Target avg: ", target_avg)
    print("Drawer: ", drawer_move_n, " | Closed drawer: ", drawer_closed_n, " | Avg drawer last: ", drawer_last)
    print("Faucet left", faucet_left_n, " | Faucet right", faucet_right_n, " | Avg faucet last: ", faucet_last)

def count(args):
    runs = [ 
        folder for folder in os.listdir(args.pwd) if 'task41' in folder and "seed10" in folder and 'reload120' in folder
    ] 
    df = pd.DataFrame(runs, columns=['runs'])
    close_drawer = []
    faucet_right = []
    faucet_left = []
    cup_forward = []
    target_dists = []
    avg_drawer_last = []
    avg_faucet_last = []
    target_avgs = []
    for run in runs:
        exp_name = os.path.join(args.pwd, run)
        print("Exp name:", exp_name)
        sv2p = 'sv2p' or 'cloning' in exp_name
        for buffer_id in range(args.buffer_num):
            try:
                if sv2p:
                    memory = os.path.join(exp_name)
                else:
                    memory = os.path.join(exp_name, 'memory',str(buffer_id))
                _count(memory, args, close_drawer, faucet_right, faucet_left, cup_forward, target_dists, avg_drawer_last, avg_faucet_last, target_avgs, sv2p=sv2p)
            except:
                print("Skipping")
                assert(False)

    assert(len(close_drawer) == len(runs))
    df['Close_drawer'] = close_drawer
    df['faucet_right'] = faucet_right
    df['faucet_left'] = faucet_left
    df['cup_forward'] = cup_forward
    df['target_dists'] = target_dists
    df['avg_drawer_last'] = avg_drawer_last
    df['avg_faucet_last'] = avg_faucet_last
    df['target_avgs'] = target_avgs
    
    df = df.sort_values('runs')
    df.to_csv(os.path.join(args.pwd, 'task_results.csv'))
        
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pwd", type=str, default="trained_models/")
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--buffer_num", type=int, default=1) 
    parser.add_argument("--num_traj_per_epoch", type=int, default=3) 
    parser.add_argument("--n_trials", type=int, default=100)
    
    args = parser.parse_args()
    
    print(args.pwd)
    count(args)
