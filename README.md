# Domain-agnostic Video Discriminator (DVD)

This code implements the following paper: 

<!-- > [Learning Generalizable Robotic Reward Functions from In-The-Wild Human Videos](https://sites.google.com/view/batch-exploration).  -->
> [Learning Generalizable Robotic Reward Functions from "In-The-Wild" Human Videos](https://sites.google.com/view/dvd-human-videos)
>
> Annie S. Chen, Suraj Nair, Chelsea Finn, 2021.

## Abstract
We are motivated by the goal of generalist robots that can complete a wide range of tasks across many environments. Critical to this is the robot's ability to acquire some metric of task success or reward, which is necessary for reinforcement learning, planning, or knowing when to ask for help. For a general-purpose robot operating in the real world, this reward function must also be able to generalize broadly across environments, tasks, and objects, while depending only on on-board sensor observations (e.g. RGB images). While deep learning on large and diverse datasets has shown promise as a path towards such generalization in computer vision and natural language, collecting high quality datasets of robotic interaction at scale remains an open challenge. In contrast, in-the-wild videos of humans (e.g. YouTube) contain an extensive collection of people doing interesting tasks across a diverse range of settings. In this work, we propose a simple approach, Domain-agnostic Video Discriminator (DVD), that learns multitask reward functions by training a discriminator to classify whether two videos are performing the same task, and can generalize by virtue of learning from a small amount of robot data with a broad dataset of human videos. We find that by leveraging diverse human datasets, this reward function (a) can generalize zero shot to unseen environments, (b) generalize zero shot to unseen tasks, and (c) can be combined with visual model predictive control to solve robotic manipulation tasks on a real WidowX200 robot in an unseen environment from a single human demo.

## Installation
1. Download the Something-Something-V2 dataset using the instructions in the original repo [here](https://github.com/TwentyBN/something-something-v2-baseline).

2. Clone this repository by running:
```
git clone https://github.com/anniesch/dvd.git
cd dvd
```
3. Install Mujoco 2.0 and mujoco-py. Instructions for this are [here](https://github.com/openai/mujoco-py#install-mujoco).

4. Create and activate conda environment with the required prerequisites:
```
conda env create -f conda_env.yml
conda activate dvd
```

5. Our simulation environment depends on Meta-World. Install it [here](https://github.com/rlworkgroup/metaworld).

6. Install the simulation env by running:
```
cd sim_env
pip install -e .
```

## Training DVD

All default args are listed in [here](https://github.com/annie268/dvd/blob/main/utils.py).

A pretrained DVD classifier trained on 3 tasks worth of robot demos and 6 tasks worth of human demos is here:
```
trained_models/dvd_human_tasks_6_robot_tasks_3.pth.tar
```

The pretrained Sth-Sth video encoder is here:
```
trained_models/video_encoder/model_best.pth.tar
```

Sample command for training: 
```
python train.py --num_tasks 6 --traj_length 0 --log_dir "test/" --similarity --batch_size 24 --im_size 120 --seed 1 --lr 0.01 --pretrained --human_data_dir [HUMAN_DATA_DIR] --sim_dir demos/ --human_tasks 5 41 44 46 93 94 --robot_tasks 5 41 93 --add_demos 60
```
All arg descriptions are located in utils.py. The ```traj_length``` arg denotes the length of video clips to train on, with 0 indicating random lengths between 20-40. The ```pretrained``` arg indicates using the pretrained Sth-Sth video encoder. The human and robot tasks to train on are indicated in a list through the ```human_tasks``` and ```robot_tasks``` args, where the numbers refer to the corresponding tasks in ```something-something-v2-labels.json```. For example, the task number 5 corresponds to "Closing something", 41 to "Moving something away from the camera", and 93 to "Moving something from left to right." ```-add_demos``` indicates training on 60 videos for each of the robot tasks.


## Using DVD for planning
We test on four different environments, each with a drawer, faucet, mug, and coffee machine. ```env1``` is the original environments, ```env2``` is with changed colors, ```env3``` is with a changed viewpoint, ```env4``` is with an altered object arrangement. 

For planning, we first train a visual dynamics model using Stochastic Variational Video Prediction (SV2P) using 10k episodes of randomly collected data. The code base can be found [here](https://github.com/tensorflow/tensor2tensor). 

To run planning with a trained SV2P model, the following is an example command, with the above DVD model:
```
python sv2p_plan.py --num_epochs 100 --num_tasks 6 --task_num 5 --seed 0 --sample_sz 100 --similarity 1 --num_demos 3 --model_dir pretrained/dvd_human_tasks_6_robot_tasks_3.pth.tar --xml env1 --cem_iters 0 --root ./ --sv2p_root [PATH TO SV2P MODEL]
```
Description of args: ```num_epochs``` is the number of planning trials, ```num_tasks``` is the number of total tasks that the DVD model was trained with, ```task_num``` denotes the task desired, so setting to 5 designates closing the drawer as the desired tasks, ```num_demos``` denotes the number of demos to randomly choose from, ```model_dir``` is the path to the trained DVD model, ```xml``` is the environment (so env1 is the original training environment), and ```cem_iters``` is the number of iterations of cem to use.

To evaluate the success rate on the tasks, run ```python analysis.py --pwd [PATH_TO_PLANNING]```.


## Using the environment:
To collect random data in an environment, use ```python collect_data.py --random --xml env1```, where ```xml``` denotes the desired environment (env1, env2, env3, or env4). 

Demos can also be collected via hard-coding by calling ```python collect_data.py --xml env1``` and altering the desired trajectory of the robot arm through ```goals``` in line 97 of the ```collect_data.py``` file. 
We provide demos in the original simulation environment used for training DVD in ```demos/```, which were collected using MPC with a ground truth reward and are very imperfect trajectories, so hard-coding demonstrations is recommended if a higher quality is needed. 



