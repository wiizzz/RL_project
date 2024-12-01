import cv2
#import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import typing as t
import vizdoom
from stable_baselines3 import ppo
from stable_baselines3.common.callbacks import EvalCallback,BaseCallback
from stable_baselines3.common import evaluation, policies
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from stable_baselines3.common import vec_env
import os
#from common import envs, plotting
# import new_env as envs
import manager_env as envs
from extractor import CustomCNN,CustomCombinedExtractor
# import new_shooting_env as envs
# import manager_env as envs
train_step =5000
def init_net(m: nn.Module):
    if len(m._modules) > 0:
       for subm in m._modules:
           init_net(m._modules[subm])
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
         nn.init.kaiming_normal_(
         m.weight, 
         a=0.1,         # Same as the leakiness parameter for LeakyReLu.
         mode='fan_in', # Preserves magnitude in the forward pass.
         nonlinearity='leaky_relu')
def init_model(model):
    init_net(model.policy)
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % 10 ==0:
            print(f'time step:{self.n_calls}')
            
        if self.n_calls % self.check_freq == 0 or self.n_calls%(train_step/2) == 0 :
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))

            self.model.save(model_path)
            print(f'model save !{self.n_calls}')

        return True
def solve_env(env_args, agent_args, n_envs, timesteps, callbacks, eval_freq=None, init_func=None):
    """Helper function to streamline the learning and evaluation process.
    
    Args:
                 env_args:    A dict containing arguments passed to the environment.
         agent_args:  A dict containing arguments passed to the agent.
         n_envs:      The number of parallel training envs to instantiate.
         timesteps:   The number of timesteps for which to train the model.
         callbacks:   A list of callbacks for the training process.
         eval_freq:   The frequency (in steps) at which to evaluate the agent.
         init_func:   A function to be applied on the agent before training.
    """
    # Create environments.
    # env = envs.create_vec_env(n_envs, **env_args)

    #env = vec_env.SubprocVecEnv([envs.make_env(**env_args) for i in range(2)])
    time_step = timesteps
    env = envs.create_vec_env(n_envs, **env_args)
    #env = vec_env.VecMonitor(env)
    # Build the agent.
    agent = ppo.PPO("MultiInputPolicy", env, tensorboard_log='logs/tensorboard',  **agent_args)
    print(agent.policy)
    # agent = agent.load('/Users/johnwei/Downloads/manager',env) 
    # Optional processing on the agent.
    # if init_func is not None: 
    #     init_func(agent)
    # Optional evaluation callback.

    log_path=f'logs/model/D3_battle/multi'
    callback = TrainAndLoggingCallback(check_freq=eval_freq, save_path=log_path) 
    callbacks.append(callback)
    if eval_freq is not None:
        #eval_env = vec_env.SubprocVecEnv([envs.make_env(**env_args) for i in range(1)])
        eval_env = envs.create_eval_vec_env(**env_args)

        callbacks.append(EvalCallback(
            eval_env, 
            n_eval_episodes=5, 
            eval_freq=eval_freq/5, 
            log_path=f'logs/evaluations/D3_battle/multi',
            best_model_save_path=f'logs/model/D3_battle/multi'))
    

            # Start the training process.
    agent.learn(total_timesteps=timesteps, tb_log_name="D3_battle", callback=callbacks)

    # Cleanup.
    env.close()
    if eval_freq is not None: eval_env.close()
    
    return agent
def frame_processor(frame):
    frame_processor = lambda frame: cv2.resize(frame[40:, 4:-4], None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    #print(frame.shape)
    if frame.shape[0] == 3:
        frame = np.moveaxis(frame,0,-1)
    new_frame = frame_processor(frame)
    #print(new_frame.shape)
    return new_frame
    
#frame_processor = lambda frame: cv2.resize(frame[40:, 4:-4], None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
env_args = {
    # 'scenario': 'D3_battle', 
    'frame_skip': 4, 
    'frame_processor': frame_processor
}

agent_args = {
    "policy_kwargs": dict(
        features_extractor_class = CustomCombinedExtractor,
    ),
    'n_epochs': 3,
    'n_steps': 16,
    'learning_rate': 1e-4,
    'batch_size': 2,
}
'''
agent_args = {
    "policy_kwargs": dict(
        features_extractor_class=CustomCombinedExtractor,
    ),
    'n_epochs': 3,
    'n_steps': 4096,
    'learning_rate': 1e-4,
    'batch_size': 32,
}
'''
def main():
    agent = solve_env(env_args, agent_args, n_envs=2, timesteps=10000, callbacks=[], eval_freq=1000,init_func = init_model)
if __name__ == "__main__":
    main()
