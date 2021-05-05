from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box
import math
import os
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from PIL import Image
from pyquaternion import Quaternion
from metaworld.envs.mujoco.utils.rotation import euler2quat
import cv2
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


class Tabletop(SawyerXYZEnv):
    def __init__(
            self,
            obj_low=None,
            obj_high=None,
            goal_low=None,
            goal_high=None,
            hand_init_pos=(0, 0.6, 0.2),
            liftThresh=0.04,
            rewMode='orig',
            rotMode='rotz',
            xml='env1',
            filepath="logs",
            max_path_length=50,
            verbose=1,
            log_freq=100, # in terms of episode num
            **kwargs
    ):
        self.max_path_length = max_path_length
        self.cur_path_length = 0
        self.xml = xml
        
        self.quick_init(locals())
        hand_low=(-0.4, 0.4, 0.0)
        hand_high=(0.4, 0.8, 0.20)
        obj_low=(-0.3, 0.4, 0.1)
        obj_high=(0.3, 0.8, 0.3)
        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./20,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )
        if obj_low is None:
            obj_low = self.hand_low

        if goal_low is None:
            goal_low = self.hand_low

        if obj_high is None:
            obj_high = self.hand_high
        
        if goal_high is None:
            goal_high = self.hand_high

        self.liftThresh = liftThresh
        self.rewMode = rewMode
        self.rotMode = rotMode
        self.hand_init_pos = np.array(hand_init_pos)
        self.action_rot_scale = 1./10
        self.action_space = Box(
                np.array([-1, -1, -1, -np.pi, -1]),
                np.array([1, 1, 1, np.pi, 1]),
            )
        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
        )

        self.imsize = 120 # im size for y axis
        self.imsize_x = int(self.imsize * 1.5) # im size for x axis
        self.observation_space = Box(0, 1.0, (self.imsize_x*self.imsize*3, ))
        self.goal_space = self.observation_space
        
        
        '''For Logging'''
        self.verbose = verbose
        if self.verbose:
            self.imgs = []
            self.filepath = filepath
            if not os.path.exists(self.filepath):
                os.mkdir(self.filepath)
        self.log_freq = log_freq
        self.epcount = -1 # num episodes so far 
        self.good_qpos = None

    @property
    def model_name(self):
        dirname = os.path.dirname(__file__)
        file = "../assets_updated/sawyer_xyz/" + self.xml + ".xml"
        filename = os.path.join(dirname, file)
        return filename

    def _get_low_dim_info(self):
        env_info =  {'mug_x': self.data.qpos[9], 
                    'mug_y': self.data.qpos[10], 
                    'mug_z': self.data.qpos[11],
                    'mug_quat_x': self.data.qpos[12], 
                    'mug_quat_y': self.data.qpos[13], 
                    'mug_quat_z': self.data.qpos[14],
                    'mug_quat_w': self.data.qpos[15],
                    'hand_x': self.get_endeff_pos()[0],
                    'hand_y': self.get_endeff_pos()[1],
                    'hand_z': self.get_endeff_pos()[2],
                    'drawer': self.data.qpos[16], 
                    'coffee_machine': self.data.qpos[17], 
                    'faucet': self.data.qpos[18], 
                    'dist': - self.compute_reward()}
        return env_info


    def step(self, action):
        
        self.set_xyz_action_rotz(action[:4])
        self.data.set_mocap_quat('mocap', np.array([0.5, 0, 0, 0.5]))
        self.do_simulation([action[-1], -action[-1]])
        self.data.qpos[12:16] = Quaternion(axis = [0,0,1], angle = 0).elements.copy()

        ob = self.get_obs()
        reward  = self.compute_reward()
        if self.cur_path_length == self.max_path_length:
            done = True
        else:
            done = False
        
        '''
        For logging
        Render images from every step if saving current episode
        '''
        if self.verbose:
            if self.epcount % self.log_freq == 0:
                im = self.sim.render(self.imsize_x, self.imsize, camera_name='cam0')
                self.imgs.append(im)

        self.cur_path_length +=1
        low_dim_info = self._get_low_dim_info()
        return ob, reward, done, low_dim_info
   
    def get_obs(self):
        obs = self.sim.render(self.imsize_x, self.imsize, camera_name="cam0") / 255.
        return obs
    
    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        start_id = 9 + self.targetobj*3
        if len(pos) < 3:
            qpos[start_id:(start_id+2)] = pos.copy()
            qvel[start_id:(start_id+2)] = 0
        else:
            qpos[start_id:(start_id+3)] = pos.copy()
            qvel[start_id:(start_id+3)] = 0
        self.set_state(qpos, qvel)
        
    def _set_obj_xyz_quat(self, pos, angle):
        quat = Quaternion(axis = [0,0,1], angle = 0).elements
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qpos[12:16] = quat.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)
          
    def initialize(self):
        self.epcount = -1 # to ensure the first episode starts with 0 idx
        self.cur_path_length = 0

    def reset_model(self, no_reset=False, add_noise=False, just_restore=False):
        ''' For logging '''
        if self.verbose and not just_restore:
            if self.epcount % self.log_freq == 0:
                # save gif of episode
                self.save_gif()
                pass
        self.cur_path_length = 0
        if not just_restore:
            self.epcount += 1
        
        if not no_reset: # reset initial block pos
            self._reset_hand(add_noise=add_noise)
            for _ in range(100):
                try:
                    self.do_simulation([0.0, 0.0])
                except:
                    print("Got Mujoco Unstable Simulation Warning")
                    continue
            self.cur_path_length = 0
            
            # Set inital pos + quat for mug
            for i in range(1):
                self.targetobj = i
                init_pos = None
                self.obj_init_pos = [0, 0.6, 0]
                if self.xml == 'updated2_view':
                    self.obj_init_pos = [0.1, 0.6, 0]
                elif self.xml == 'updated2_view2' or self.xml == 'updated2_view3':
                    self.obj_init_pos = [0.11, 0.6, 0]
                self._set_obj_xyz_quat(self.obj_init_pos, [0])
        
            # Set initial pos for drawer, coffee machine button, and faucet
            self.data.qpos[16] = -0.07
            self.data.qpos[17] = 0
            self.data.qpos[18] = 0

        self.sim.forward()
        o = self.get_obs()
        if self.epcount % self.log_freq == 0 and not just_restore:
            self.imgs = []
            im = self.sim.render(self.imsize_x, self.imsize, camera_name='cam0')
            self.imgs.append(im)
        low_dim_info = self._get_low_dim_info()
        return o, low_dim_info 


    def _reset_hand(self, pos=None, add_noise=False):
        if self.epcount < 2 and self.cur_path_length == 0:
            self.good_qpos = self.sim.data.qpos[:7].copy()
        self.data.qpos[:7] = self.good_qpos
        if pos is None:
            pos = [0, 0.5, 0.02]
            if add_noise:
                np.random.uniform(-0.02, 0.02, (3,)) 
        for _ in range(100):
            self.data.set_mocap_pos('mocap', pos)
            self.data.set_mocap_quat('mocap', np.array([0.5, 0, 0, 0.5]))
            try:
                self.do_simulation([-1,1], self.frame_skip)
            except:
                print("Got Mujoco Unstable Simulation Warning")
                continue
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_reward(self):
        return 0.0
    
    def set_qpos(self, qpos):
        for _ in range(100):
            try:
                self.do_simulation([-1,1], self.frame_skip)
            except:
                print("Got Mujoco Unstable Simulation Warning")
                continue
        self.sim.forward()
        o = self.get_obs()
        return o
    
    def take_steps_and_render(self, obs, actions, set_qpos=None):
        '''Returns image after having taken actions from obs.'''
        # Set starting position
        threshold = 0.05
        if set_qpos is not None:
            self.data.qpos[:] = set_qpos.copy()
        else:
            self.reset_model()
            # Set inital pos + quat for mug
            for i in range(1):
                self.targetobj = i
                init_pos = None
                self.obj_init_pos = obs[(i+1)*3:((i+1)*3)+3]     
                self.obj_init_quat =  obs[(i+1)*3+3:((i+1)*3)+7]  
                self._set_obj_xyz_quat(self.obj_init_pos, self.obj_init_quat)

            # Set initial pos for drawer, coffee machine button, and faucet
            self.data.qpos[16] = obs[-3]
            self.data.qpos[17] = obs[-2]
            self.data.qpos[18] = obs[-1]
            self.sim.forward()
        self._reset_hand(pos=obs[:3])
        
        # Get trajectory
        imgs = []
        im = self.sim.render(self.imsize_x, self.imsize, camera_name='cam0') / 255.0
        imgs.append(im)
        ''' Then take the selected actions '''
        for i in range(actions.shape[0]):
            action = actions[i]
            self.set_xyz_action_rotz(action[:4])
            self.data.set_mocap_quat('mocap', np.array([0.5, 0, 0, 0.5]))
            self.do_simulation([action[-1], -action[-1]])
            self.data.qpos[12:16] = Quaternion(axis = [0,0,1], angle = 0).elements.copy()
            im = self.sim.render(self.imsize_x, self.imsize, camera_name='cam0') / 255.0
            imgs.append(im)
        return imgs
        
    def _restore(self):
        '''For resetting the env without having to call reset() (i.e. without updating episode count)'''
        self.reset_model(just_restore=True)

    def save_goal_img(self, goal, actions=None, angle=None):
        '''Returns image with a given goal array of positions for the gripper and blocks.'''
        self._reset_hand(pos=goal[:3])

        #  Move objects to correct positions
        for i in range(1):
            self.targetobj = i
            init_pos = None
            self.obj_init_pos = goal[(i+1)*3:((i+1)*3)+3]                
            self._set_obj_xyz_quat(self.obj_init_pos, [0])
            ("set obj pos", i)
        
        if angle is not None:
            print("angle", angle)
            self.data.qpos[16] = angle
            self.data.qpos[17] = angle
            self.data.qpos[18] = angle
        self.sim.forward()
        im = self.sim.render(self.imsize_x, self.imsize, camera_name='cam0')
        return im

    def move_gripper(self, ob, goal, n_steps):
        '''Move end effector (ob) to certain place (goal) by n_steps action steps'''
        imgs = []
        actions = []
        goal = goal #[-0.1, 0.55, 0]  
        gripper = self.get_endeff_pos()
        action_total = np.concatenate([(goal-gripper), np.array([np.random.uniform(-np.pi, np.pi), -1])])
        im = self.sim.render(self.imsize_x, self.imsize, camera_name='cam0') / 255.0
        imgs.append(im)
        for i in range(n_steps):
            next_action = action_total
            actions.append(next_action)
            ob, reward, done, low_dim_info = self.step(next_action)
            imgs.append(ob)
        return imgs, actions, low_dim_info
            
    
    def save_gif(self):
        ''' Saves the gif of an episode '''
        try: 
            with imageio.get_writer(
                    self.filepath + '/Eps' + str(self.epcount) + '.gif', mode='I') as writer:
                for i in range(self.max_path_length + 1):
                    writer.append_data(self.imgs[i])
        except:
            pass

