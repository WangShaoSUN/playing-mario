import numpy as np
from monitor import Monitor
import gym
from gym import spaces
from collections import  deque
from multiprocessing import  Process,Pipe
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from abc import ABC,abstractmethod
print("gym version:",gym.__version__)
import matplotlib.pyplot as plt
import cv2

class MaxAndSkip(gym.Wrapper):
    def __init__(self,env,skip=4):
        super(MaxAndSkip,self).__init__(env)
        self.__obs_buffer=deque(maxlen=2)
        self.__skip=skip
    def step(self,action):
        total_reward=0.
        done=None
        for i in range(self.__skip):
            obs,reward,done,info=self.env.step(action)
            self.__obs_buffer.append(obs)
            total_reward+=reward
            if done:
                break
        max_frame=np.max(np.stack(self.__obs_buffer),axis=0)
        return max_frame,total_reward,done,info
    def reset(self):
        self.__obs_buffer.clear()
        obs=self.env.reset()
        self.__obs_buffer.append(obs)
        return obs
class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self,env):
        super(ProcessFrame84,self).__init__(env)
        self.observation_space=spaces.Box(low=0,high=256,shape=(84,84,1))

    def observation(self, observation):
        # print("dd:",observation.size)
        if observation.size==184320:
             img=np.reshape(observation,[240,256,3]).astype(np.float32)

        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[16:100, :]
        x_t=np.reshape(x_t,[84,84,1])

        return x_t.astype(np.uint8)
    @staticmethod
    def prcess(frame):
         pass
class ImageToPytorch(gym.ObservationWrapper):
    def __init__(self,env):
        super(ImageToPytorch,self).__init__(env)
        old_shape=self.observation_space.shape
        # print("old shape:",old_shape)
        self.observation_space=gym.spaces.Box(low=0.,high=1.0,shape=(old_shape[-1],
                                              old_shape[0],old_shape[1]))

    def observation(self, observation):
        return np.transpose(observation,(2,0,1))
class ClipReward(gym.RewardWrapper):
     def reward(self,reward):
         return np.sign(reward)

class LazyFrames(object):
    def __init__(self,frames):
        self._frame=frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frame, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

class FrameStack(gym.Wrapper):
    def __init__(self,env,k):
        " stack the last k frames"
        # gym.wrappers.__init__(self,env)
        super(FrameStack,self).__init__(env)
        self.k=k
        self.frames=deque([],maxlen=k)
        shape=env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shape[0] * k, shape[1], shape[2]))

    def reset(self):
        obs=self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info
    def _get_obs(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames)).__array__(np.float)

class  VecEnv(ABC):
    def __init__(self,num_envs,observation_space,action_space):
        self.num_envs=num_envs
        self.observation_space=observation_space
        self.action_space=action_space

    @abstractmethod
    def reset(self):
        pass
    @abstractmethod
    def step_async(self,action):
        pass
    @abstractmethod
    def step_wait(self):
        pass
    @abstractmethod
    def close(self):
        pass
    def step(self,actions):
        self.step_async(actions)
        return self.step_wait()
    def render(self):
        pass
class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
def worker(remote,paraent_remote,env):
    paraent_remote.close()
    env = env.x()
    while True:
        cmd,data=remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
              ob = env.reset()
            # env.render()
            remote.send((ob, reward, done, info))
        elif cmd  == 'reset':

            ob = env.reset()

            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError
class SubprocVecEnv(VecEnv):
    def __init__(self,env_fns,space=None):
        self.waiting = False
        self.closed  = False

        len_env=len(env_fns)

        self.remote,self.work_remote=zip(*[Pipe() for i in range(len_env)])
        self.ps=[Process(target=worker,args=(work_remote,remote,CloudpickleWrapper(env)))
                   for (work_remote,remote, env) in zip(self.work_remote,self.remote,env_fns)]

        for p in self.ps:
            p.daemon= True
            p.start()
        for remote in self.work_remote:
            remote.close()
        self.remote[0].send(('get_spaces', None))
        observation_space, action_space = self.remote[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)


    def  step_async(self,actions):
        for remote,action in zip(self.remote,actions):
            remote.send(("step",action))
        self.waiting=True
    def step_wait(self):
        results = [remote.recv() for remote in self.remote]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos
    def reset(self):
        for remote in self.remote:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remote])
    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remote])
    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
def wrap_cover(env_name,seed):
    def wrap_():
        env=gym_super_mario_bros.make(env_name)
        env = JoypadSpace(env, [["right"], ["right", "A"]])
        env.seed(seed)
        env = Monitor(env, './')
        env = MaxAndSkip(env, skip=4)
        env = ProcessFrame84(env)
        env = ImageToPytorch(env)



        env = FrameStack(env, 4)



        env = ClipReward(env)
        return env
    return wrap_
if __name__ == '__main__':
    # env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    # env = ProcessFrame84(env)
    # env = ImageToPytorch(env)
    # env.reset()
    # next_state, reward, done, info = env.step(action=0)
    # # print(next_state.shape)
    # wrap_cover("dd",3)
    # plt.imshow(next_state[0])
    # plt.show()
    # env=ProcessFrame84(env)
    # env = ProcessFrame84(env)



    env = SubprocVecEnv([wrap_cover("d", 3) for i in range(5)])
    print(env.action_space.n)
    print(env.observation_space.shape)
    s=np.array(env.reset())
    print(s.shape)
