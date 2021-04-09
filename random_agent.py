import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from replay_memory import ReplayBuffer
from logx import EpochLogger
from logx import setup_logger_kwargs
import random
import os
import pickle
import time
from collections import deque
import matplotlib.pyplot as plt
from env_wrapper import  wrap_cover, SubprocVecEnv
from mvg_ import *
import argparse
parser = argparse.ArgumentParser(description='Some settings of the experiment.')
parser.add_argument('--games', type=str,default="SuperMarioBros-1-2-v0", help='name of the games. for example: Breakout')
parser.add_argument('--seed', type=int,default=10, help='seed of the games')
parser.add_argument('--n_env', type=int,default=32, help='seed of the games')
args = parser.parse_args()
args.games = "".join(args.games)

STATE_LEN = 4
# target policy sync interval
TARGET_REPLACE_ITER = 1
# simulator steps for start learning
LEARN_START = int(1e+3)
# (prioritized) experience replay memory size
MEMORY_CAPACITY = int(5e+5)
# simulator steps for learning interval
LEARN_FREQ = 4

N_ENVS = args.n_env
# Total simulation step
STEP_NUM = int((2e+7)+2)
# gamma for MDP
GAMMA = 0.99
# visualize for agent playing
RENDERING = False
# openai gym env name
ENV_NAME = args.games
env = SubprocVecEnv([wrap_cover(ENV_NAME,args.seed+i) for i in range(N_ENVS)])
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape

USE_GPU = torch.cuda.is_available()
print('USE GPU: '+str(USE_GPU))
# mini-batch size
BATCH_SIZE = 64
# learning rage
LR = 1e-4
# epsilon-greedy
EPSILON = 1.0

SAVE = True
LOAD = False
# save frequency
SAVE_FREQ = int(1e+3)
# paths for predction net, target net, result log
PRED_PATH = './data/model/dqn_pred_net_o_'+args.games+'.pkl'
TARGET_PATH = './data/model/dqn_target_net_o_'+args.games+'.pkl'
RESULT_PATH = './data/plots/dqn_result_o_'+args.games+'.pkl'




class random_agent(object):
    def __init__(self):
        pass
        
        
    def choose_action(self, x, EPSILON):
        # x:state
        

      
            # random exploration case
        action = np.random.randint(0, N_ACTIONS, (x.shape[0]))
        return action

    

   


dqn = random_agent()
logdir = './random_agent/%s' % args.games + '/%i' % int(time.time())

logger_kwargs = setup_logger_kwargs(args.games, args.seed, data_dir=logdir)
logger = EpochLogger(**logger_kwargs)
kwargs = {

    'seed': args.seed,
}
logger.save_config(kwargs)
# model load with check


print('Collecting experience...')

# episode step for accumulate reward
epinfobuf = deque(maxlen=100)
# check learning time
start_time = time.time()

# env reset
s = np.array(env.reset())
print("s.shape:",s.shape)

# for step in tqdm(range(1, STEP_NUM//N_ENVS+1)):
for step in range(1, STEP_NUM // N_ENVS + 1):
    a = dqn.choose_action(s, EPSILON)
    # print('a',a)

    # take action and get next state
    s_, r, done, infos = env.step(a)
    # log arrange
    for info in infos:
        maybeepinfo = info.get('episode')
        if maybeepinfo: epinfobuf.append(maybeepinfo)
    # print log and save
    if step % SAVE_FREQ == 0:
        # check time interval
        time_interval = round(time.time() - start_time, 2)
        period_results = [epinfo['r'] for epinfo in epinfobuf]
        # calc mean return
        mean_100_ep_return = round(np.mean([epinfo['r'] for epinfo in epinfobuf]), 2)
#         result.append(mean_100_ep_return)
        logger.log_tabular('Epoch', t // steps_per_epoch)
        # print log
        
#         logger.log_tabular('TotalEnvInteracts', dqn.memory_counter)
        logger.log_tabular('AverageEpRet', mean_100_ep_return)
        logger.log_tabular('MinEpRet', np.min(period_results))
        logger.log_tabular('MaxEpRet', np.max(period_results))
        logger.log_tabular('time', time_interval)
       
        logger.dump_tabular()
#         save model
#         dqn.save_model()
        # pkl_file = open(RESULT_PATH, 'wb')
        # pickle.dump(np.array(result), pkl_file)
        # pkl_file.close()

    s = s_
    if RENDERING:
        env.render()
print("The training is done!")
