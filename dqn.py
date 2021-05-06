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
parser.add_argument('--n_env', type=int,default=16, help='seed of the games')
parser.add_argument('--lr', type=float,default=0.0001, help='seed of the games')
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
LR = args.lr
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

import os.path as osp
if osp.exists("./data/model"):
    print(" directory exist")
else:
    os.makedirs("./data/model")
    
class NoisyLinear(nn.Linear):
  def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
    super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
    # µ^w and µ^b reuse self.weight and self.bias
    self.sigma_init = sigma_init
    self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
    self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
    self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
    self.register_buffer('epsilon_bias', torch.zeros(out_features))
    self.reset_parameters()

  def reset_parameters(self):
    if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
      init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.constant(self.sigma_weight, self.sigma_init)
      init.constant(self.sigma_bias, self.sigma_init)

  def forward(self, input):
    return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), self.bias + self.sigma_bias * Variable(self.epsilon_bias))

  def sample_noise(self):
    self.epsilon_weight = torch.randn(self.out_features, self.in_features)
    self.epsilon_bias = torch.randn(self.out_features)

  def remove_noise(self):
    self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
    self.epsilon_bias = torch.zeros(self.out_features)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.feature_extraction = nn.Sequential(
            # Conv2d(输入channels, 输出channels, kernel_size, stride)
            nn.Conv2d(STATE_LEN, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(7 * 7 * 64, 512)
#         self.fc = NoisyLayer_with_MVG(7 * 7 * 64, 512)
        # action value
        self.fc_q = nn.Linear(512, N_ACTIONS)
#         self.fc_q = NoisyLayer_with_MVG(512, N_ACTIONS)
        # 初始化参数值
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x.size(0) : minibatch size
        mb_size = x.size(0)
        x = self.feature_extraction(x / 255.0)  # (m, 7 * 7 * 64)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        action_value = self.fc_q(x)

        return action_value

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))


class DQN(object):
    def __init__(self):
        self.pred_net, self.target_net = ConvNet(), ConvNet()
        # sync evac target
        self.update_target(self.target_net, self.pred_net, 1.0)
        # use gpu
        if USE_GPU:
            self.pred_net.cuda()
            self.target_net.cuda()

        # simulator step counter
        self.memory_counter = 0
        # target network step counter
        self.learn_step_counter = 0
        # loss function
        self.loss_function = nn.MSELoss()
        # ceate the replay buffer
        self.replay_buffer = ReplayBuffer(MEMORY_CAPACITY)

        # define optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=LR)

    def update_target(self, target, pred, update_rate):
        # update target network parameters using predcition network
        for target_param, pred_param in zip(target.parameters(), pred.parameters()):
            target_param.data.copy_((1.0 - update_rate) \
                                    * target_param.data + update_rate * pred_param.data)

    def save_model(self):
        # save prediction network and target network
        self.pred_net.save(PRED_PATH)
        self.target_net.save(TARGET_PATH)

    def load_model(self):
        # load prediction network and target network
        self.pred_net.load(PRED_PATH)
        self.target_net.load(TARGET_PATH)

    def choose_action(self, x, EPSILON):
        # x:state
        x = torch.FloatTensor(x)
        # print(x.shape)
        if USE_GPU:
            x = x.cuda()

        # epsilon-greedy策略
        if np.random.uniform() >= EPSILON:
            # greedy case
            action_value = self.pred_net(x)  # (N_ENVS, N_ACTIONS, N_QUANT)
            action = torch.argmax(action_value, dim=1).data.cpu().numpy()
        else:
            # random exploration case
            action = np.random.randint(0, N_ACTIONS, (x.size(0)))
        return action

    def store_transition(self, s, a, r, s_, done):
        self.memory_counter += 1
        self.replay_buffer.add(s, a, r, s_, float(done))

    def learn(self):
        self.learn_step_counter += 1
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.update_target(self.target_net, self.pred_net, 1e-2)

        b_s, b_a, b_r, b_s_, b_d = self.replay_buffer.sample(BATCH_SIZE)
        # b_w, b_idxes = np.ones_like(b_r), None

        b_s = torch.FloatTensor(b_s)
        b_a = torch.LongTensor(b_a)
        b_r = torch.FloatTensor(b_r)
        b_s_ = torch.FloatTensor(b_s_)
        b_d = torch.FloatTensor(b_d)

        if USE_GPU:
            b_s, b_a, b_r, b_s_, b_d = b_s.cuda(), b_a.cuda(), b_r.cuda(), b_s_.cuda(), b_d.cuda()

        # action value for current state
        q_eval = self.pred_net(b_s)
        mb_size = q_eval.size(0)
        q_eval = torch.stack([q_eval[i][b_a[i]] for i in range(mb_size)])

        # optimal action value for current state
        q_next = self.target_net(b_s_)
        # best_actions = q_next.argmax(dim=1)
        # q_next = torch.stack([q_next[i][best_actions[i]] for i in range(mb_size)])
        q_next = torch.max(q_next, -1)[0]
        q_target = b_r + GAMMA * (1. - b_d) * q_next
        q_target = q_target.detach()

        # loss
        loss = self.loss_function(q_eval, q_target)
        logger.store(loss=loss)
        # backprop loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


dqn = DQN()
logdir = './Noisy_DQN/%s' % args.games + '/%i' % int(time.time())

logger_kwargs = setup_logger_kwargs(args.games, args.seed, data_dir=logdir)
logger = EpochLogger(**logger_kwargs)
kwargs = {

    'seed': args.seed,
    'lr"args.lr,
}
logger.save_config(kwargs)
# model load with check
if LOAD and os.path.isfile(PRED_PATH) and os.path.isfile(TARGET_PATH):
    dqn.load_model()
    pkl_file = open(RESULT_PATH, 'rb')
    result = pickle.load(pkl_file)
    pkl_file.close()
    print('Load complete!')
else:
    result = []
    print('Initialize results!')

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
    s_ = np.array(s_)

    # clip rewards for numerical stability
    clip_r = np.sign(r)

    # store the transition
    for i in range(N_ENVS):
        dqn.store_transition(s[i], a[i], clip_r[i], s_[i], done[i])

    # annealing the epsilon(exploration strategy)
    if step <= int(1e+4):
        # linear annealing to 0.9 until million step
        EPSILON -= 0.9 / 1e+4
    elif step <= int(2e+4):
        # else:
        # linear annealing to 0.99 until the end
        EPSILON -= 0.09 / 1e+4

    # if memory fill 50K and mod 4 = 0(for speed issue), learn pred net
    if (LEARN_START <= dqn.memory_counter) and (dqn.memory_counter % LEARN_FREQ == 0):
        loss = dqn.learn()

    # print log and save
    if step % SAVE_FREQ == 0:
        # check time interval
        time_interval = round(time.time() - start_time, 2)
        period_results = [epinfo['r'] for epinfo in epinfobuf]
        # calc mean return
        mean_100_ep_return = round(np.mean([epinfo['r'] for epinfo in epinfobuf]), 2)
        result.append(mean_100_ep_return)
        # logger.log_tabular('Epoch', t // steps_per_epoch)
        # print log
        print('Used Step: ', dqn.memory_counter,
              '| EPS: ', round(EPSILON, 3),
              # '| Loss: ', loss,
              '| Mean ep 100 return: ', mean_100_ep_return,
              '| Used Time:', time_interval)
        logger.log_tabular('TotalEnvInteracts', dqn.memory_counter)
        logger.log_tabular('AverageEpRet', mean_100_ep_return)
        logger.log_tabular('MinEpRet', np.min(period_results))
        logger.log_tabular('MaxEpRet', np.max(period_results))
        logger.log_tabular('time', time_interval)
        logger.log_tabular("loss", with_min_and_max=True)
        logger.dump_tabular()
#         save model
        #dqn.save_model()
        # pkl_file = open(RESULT_PATH, 'wb')
        # pickle.dump(np.array(result), pkl_file)
        # pkl_file.close()

    s = s_
    if RENDERING:
        env.render()
print("The training is done!")
