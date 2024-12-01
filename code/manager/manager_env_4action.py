import typing as t

import numpy as np
import vizdoom
from gymnasium import Env
from gymnasium import spaces
from stable_baselines3.common import vec_env
from action_combine import *
from stable_baselines3 import PPO
from stable_baselines3 import ppo
Frame = np.ndarray
from extractor import CustomCNN
from typing import Callable
from collections import OrderedDict
import pickle
import json


class DoomEnv(Env):
    """Wrapper environment following OpenAI's gym interface for a VizDoom game instance."""

    def __init__(self,game: vizdoom.DoomGame,frame_processor: t.Callable,frame_skip: int = 4,evl:bool = False):
        super().__init__()

        # Determine action space
        self.game = game
        '''
        self.possible_actions = get_available_actions(np.array([
             Button.MOVE_FORWARD, Button.MOVE_RIGHT, 
             Button.MOVE_LEFT, Button.TURN_LEFT, Button.TURN_RIGHT]))
        '''
        self.action_space = spaces.Discrete(4)

        #read labels data from the files
        self.lb = []
        labels_path = "prompts and labels/labels01_resize/"
        for i in range(100):       
            f0 = open(labels_path+f"{i}.pkl","rb")
            label = pickle.load(f0)
            label = np.array(label)
            #print(label)
            self.lb.append(label)

        prompt_file_path = 'prompts and labels/second_prompt_action_all.json'

        # Read prompt data from the file
        with open(prompt_file_path, 'r') as file:
            data = json.load(file)
        self.prompts = []
        for dic in data:
            self.prompts.append(dic["action_list"])

        # Determine observation space
        h, w, c = game.get_screen_height(), game.get_screen_width(), game.get_screen_channels()
        new_h, new_w, new_c = frame_processor(np.zeros((h, w, c))).shape
        self.img_space = spaces.Box(low=0, high=255, shape=(new_h, new_w, new_c), dtype=np.uint8)
        # self.observation_space = OrderedDict()
        # self.obseation_space["image"] = spaces.Box(low=0, high=255, shape=(new_h, new_w, new_c), dtype=np.uint8)
        # self.observation_space["vector"] = spaces.Box(low=0, high=255, shape=(2,), dtype=np.uint8)
        # self.observation_space = spaces.Dict(self.observation_space)
        self.observation_space = spaces.Dict(
            {"image" : spaces.Box(low=0, high=255, shape=(new_h, new_w, new_c), dtype=np.uint8),
            "vector" : spaces.Box(low=0, high=255, shape=(2,), dtype=np.uint8)}
        )
        #self.observation_space = spaces.Box(low=0, high=255, shape=(new_h, new_w, new_c), dtype=np.uint8)
        # self.observation_space = spaces.Dict({"image": spaces.Box(low=0, high=255, shape=(new_h, new_w, new_c), dtype=np.uint8), "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.uint8)})
        # for key,values in  self.observation_space.items():
        #     print("hi", key)
        
        # Assign other variables
        self.item_count = 0
        #self.possible_actions = np.eye(self.action_space.n).tolist()  # VizDoom needs a list of buttons states.
        self.frame_skip = frame_skip
        self.frame_processor = frame_processor
        self.init_x, self.init_y = self._get_player_pos()
        self.last_x, self.last_y = self._get_player_pos()
        self.step_count = 0
        self.health = self.game.get_game_variable(vizdoom.GameVariable.HEALTH)
        self.empty_frame = OrderedDict()
        self.empty_frame["image"] = np.zeros(self.img_space.shape, dtype=np.uint8)
        self.empty_frame["vector"] = [0,0]
        #self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)
        # self.empty_frame = spaces.Dict(self.empty_frame)
        # self.empty_frame = spaces.Dict({"image": np.zeros(self.img_space.shape, dtype=np.uint8), "vector": [0,0]})
        self.state = self.empty_frame
        self.nav_agent = ppo.PPO.load('model/navigation.zip', custom_objects={'policy_kwargs': {'features_extractor_class': CustomCNN}})
        self.shooting_agent = ppo.PPO.load('model/shooting.zip', custom_objects={'policy_kwargs': {'features_extractor_class': CustomCNN}})
        self.ammo_agent = ppo.PPO.load('model/ammo.zip', custom_objects={'policy_kwargs': {'features_extractor_class': CustomCNN}})
        self.health_agent = ppo.PPO.load('model/health.zip', custom_objects={'policy_kwargs': {'features_extractor_class': CustomCNN}})
        self.nav_actions = get_available_actions(np.array([
             Button.MOVE_FORWARD, Button.MOVE_RIGHT, 
             Button.MOVE_LEFT, Button.TURN_LEFT, Button.TURN_RIGHT]))
        self.shooting_actions = get_available_actions(np.array([
             Button.MOVE_FORWARD, Button.MOVE_RIGHT, 
             Button.MOVE_LEFT, Button.TURN_LEFT, Button.TURN_RIGHT, Button.ATTACK]))
        self.ammo_actions = self.nav_actions
        self.health_actions = self.nav_actions
        self.skill_base = [self.navigation,self.shooting,self.health,self.ammo,
                lambda:self.take_cover(90),lambda:self.take_cover(60),lambda:self.take_cover(30),lambda:self.take_cover(0),lambda:self.take_cover(-30),
                lambda:self.take_cover(-60),lambda:self.take_cover(-90)]
        self.kill_count =0
        self.w = 0
        self.evl = evl
        self.record = []
        self.steps = 0
        self.prompt_done = -1
        self.target_prompt = None
        self.prompt_gamma = 0.01
        self.prompt_gamma_o = self.prompt_gamma
        self.decay_epoch = 100


    def step(self, action: int,) -> t.Tuple[Frame, int, bool, t.Dict]:
        #self.prompt_gamma = max(self.prompt_gamma - 1e-5,0)
        self.step_count += 1
        self.recording()
        total_reward = 0
        prompt_reward = 0
        
        if self.game.get_state() is not None:
            state = self.game.get_state()
            slabels = state.labels
            Imp_count=0
            for label in slabels:
                if label.object_name == 'DoomImp' and label.value >= 50:
                    Imp_count += 1
            if Imp_count >= 0 :     ### if last prompt is evaluated and there is enemy in vision --> prompt
                self.prompt_done = 0
                label_goal = state.labels_buffer
                target_label = self.classify(label_goal)
                self.target_prompt = self.prompts[target_label]
                prompt_reward = self.prompt_reward(self.prompt_done,self.target_prompt,action)
                self.prompt_done += 1

        #skill = self.skill_base[action]
        #reward,self.state,done = self.skill_base[action]()
        
        match action:
            case 0:
                reward,self.state,done = self.shooting()
            case 1:
                reward,self.state,done = self.navigation()
            case 2:
                reward,self.state,done = self.ammo()
            case 3:
                reward,self.state,done = self.health_gathering()
        '''
        match action:
            case 0:
                #print(0)
                reward,self.state,done = self.shooting()
            case 1:
                #print(1)
                reward,self.state,done = self.take_cover(-90)
            case 2:
                #print(2)
                reward,self.state,done = self.take_cover(-60)        
            case 3:
                #print(3)
                reward,self.state,done = self.take_cover(-30)       
            case 4:
                #print(4)
                reward,self.state,done = self.take_cover(0)       
            case 5:
                #print(5)
                reward,self.state,done = self.take_cover(30)
            case 6:
                #print(6)
                reward,self.state,done = self.take_cover(60)
            case 7:
                #print(7)
                reward,self.state,done = self.take_cover(90)
            case 8:
                #print(8)
                reward,self.state,done = self.health_gathering()
            case 9:
                #print(9)
                reward,self.state,done = self.ammo()
            case 10:
                #print(10)
                reward,self.state,done = self.navigation()
        '''
        #self.skill_base[?]
        # if action == 0:
        #     reward,self.state,done = self.navigation()
        # elif action == 1:
        #     reward,self.state,done = self.shooting()
        '''    
        print(self.w%10)
        print(f'action:{action}')
        print(f'reward:{reward}')
        '''
        info ={'info':[]}
        if done and self.evl:
            info = {'info':self.record}
        #self.state = self._get_frame()
        # if()
        # print("=========nit==============\n",[self.game.get_game_variable(vizdoom.GameVariable.HEALTH), self.game.get_game_variable(vizdoom.GameVariable.AMMO2)])
        if done:
            self.state = self.empty_frame
            #print('done')
        else:
            self.state = {"image": self.state, "vector": [self.game.get_game_variable(vizdoom.GameVariable.HEALTH), self.game.get_game_variable(vizdoom.GameVariable.AMMO2)]}
        total_reward = reward + prompt_reward
        return self.state, total_reward, done, False, info
    


    def prompt_reward(self,prompt_done,prompt,action):
        reward = 0
        take_cover_dir = 0
        prompt_done = 0
        i = 0
        while prompt[i]==2:
            i+=1
        if i >= 5:
            return 0
        if prompt[i]==1 and action == 0:
            reward += self.prompt_gamma
        elif action == 1 and prompt[i]==5:
            reward += self.prompt_gamma
        elif action == 2 and prompt[i]==4:
            reward += self.prompt_gamma
        elif action == 3 and prompt[i]==3:
            reward += self.prompt_gamma
            
        '''
        if(prompt[prompt_done]==2):
            take_cover_dir = (prompt[prompt_done+5]-90)/30
            if (action > 0 and action < 8):
                reward += self.prompt_gamma/(abs((action-4)-take_cover_dir)+1)
                return reward                
        else:
            if action == 0 and prompt[prompt_done] == 1:
                reward = self.prompt_gamma
            elif action == 8 and prompt[prompt_done] == 3:
                reward = self.prompt_gamma
            elif action == 9 and prompt[prompt_done] == 4:
                reward = self.prompt_gamma
            elif action == 10 and prompt[prompt_done] == 5:
                reward = self.prompt_gamma
            return reward 
        '''    
        return reward

    def classify(self,label_goal):
        p = np.full(100, 10**6)
        label_goal[label_goal > 1] = 0
        label_goal[label_goal < 0] = 0
        for i in range(100):
            #print(np.sum(labels[i]))
            p[i]=np.sum(np.abs(label_goal-self.lb[i]))   

        return p.argmin()
    
    def recording(self):
        if not self.evl:
            return
        if self.game.is_episode_finished():
            return
        else:
            state_o = self.game.get_state().screen_buffer
            self.record.append(np.moveaxis(state_o,0,-1))

    def navigation(self,evl=False):
        reward = 0
        self.state = self._get_frame()
        action,_ = self.nav_agent.predict(self.state)
        reward += self.game.make_action(self.nav_actions[action],4)
        reward += self.reward(action)
        self.recording()
        while True:
            done = self.game.is_episode_finished()
            self.state = self._get_frame(done)
            if done :
                return reward, self.state, done

            state = self.game.get_state()
            labels = state.labels
            Imp_count=0
            for label in labels:
                if label.object_name == 'DoomImp' and label.value >= 200:
                    Imp_count += 1
            if Imp_count > 0 :
                #print('kkkk')
                return reward, self.state, done
            else:
                action,_ = self.nav_agent.predict(self.state)
                reward += self.game.make_action(self.nav_actions[action],4)
                self.recording()
                reward += self.reward(action)
    def shooting(self):
        reward = 0
        no_e_count = 0
        first = True 
        while True:
            done = self.game.is_episode_finished()
            self.state = self._get_frame(done)
            if done :
                return reward, self.state, done
            
            state = self.game.get_state()
            labels = state.labels
            Imp_count=0
            for label in labels:
                if label.object_name == 'DoomImp' and label.value >= 200:
                    Imp_count += 1
            no_e_count += 1 if Imp_count == 0 else 0
            no_e_count = 0 if Imp_count>0 else no_e_count
            if Imp_count>0:
                first = False
            if Imp_count > 0 or no_e_count <=3 :
                action,_ = self.shooting_agent.predict(self.state)
                reward += self.game.make_action(self.shooting_actions[action],4)
                self.recording()
                reward += self.reward(action) 
            else:
                if first:
                    reward -= 0.1
                return reward, self.state, done
    def ammo(self):
        ammo_o = self.game.get_game_variable(vizdoom.GameVariable.AMMO2)
        reward = 0
        step_count = 0
        while step_count<200:
            done = self.game.is_episode_finished()
            self.state = self._get_frame(done)
            if done :
                return reward, self.state, done
            
            state = self.game.get_state()
            labels = state.labels
            Imp_count=0
            for label in labels:
                if label.object_name == 'DoomImp' and label.value >= 200:
                    Imp_count += 1
            ammo = self.game.get_game_variable(vizdoom.GameVariable.AMMO2)
            if  ammo <= ammo_o:
                action,_ = self.ammo_agent.predict(self.state)
                reward += self.game.make_action(self.ammo_actions[action],4)
                self.recording()
                reward += self.reward(action) 
                if Imp_count > 0 :
                    return reward, self.state, done
            else:
                return reward, self.state, done
        return reward, self.state, done
        
    def health_gathering(self):
        health_o = self.game.get_game_variable(vizdoom.GameVariable.HEALTH)
        reward = 0
        step_count = 0
        while step_count<200:
            done = self.game.is_episode_finished()
            self.state = self._get_frame(done)
            if done :
                return reward, self.state, done
            
            state = self.game.get_state()
            labels = state.labels
            Imp_count=0
            for label in labels:
                if label.object_name == 'DoomImp' and label.value >= 200:
                    Imp_count += 1
            health = self.game.get_game_variable(vizdoom.GameVariable.HEALTH)
            if  health <= health_o:
                action,_ = self.health_agent.predict(self.state)
                reward += self.game.make_action(self.health_actions[action],4)
                self.recording()
                reward += self.reward(action) 
                if Imp_count > 0 :
                    #print('kkkk')
                    return reward, self.state, done
            else:
                return reward, self.state, done
        return reward, self.state, done
 
    def wall_ratio(self):
        wall = 0
        if self.game.get_state() is not None :
            labels_buffer = self.game.get_state().labels_buffer
            wall = np.count_nonzero(labels_buffer==0)
        return wall/(240*320)

    def take_cover(self,turn):
        reward,done = self.turn_angle(turn)
        if done:
            self.state = self._get_frame(done)
            # print("exit_take_cover")
            return reward,self.state,done
        step_count = 0
        step_limit = 100
        #while np.average(depth_buffer[230:250,310:330])>5:
        while self.wall_ratio()<0.9 and step_count <= step_limit:
            # print("take_cover")
            reward += self.game.make_action([1,0,0,0,0,0])
            
            self.recording()
            step_count +=1
            if self.game.is_episode_finished():
                self.state = self._get_frame(True)
                return reward,self.state ,True
        self.state = self._get_frame(done)
        # print("exit_take_cover")
        return reward, self.state ,False

    def turn_angle(self,turn):
        # turn left:positive
        angle_o = self.game.get_game_variable(vizdoom.GameVariable.ANGLE)
        target = angle_o + turn
        cross_zero = target >= 360 or target<0
        target = target-360 if target>=360 else target
        target = 360-target if target < 0 else target
        angle_l = angle_o
        reward = 0
        while True:
            # print("turn_angle")
            angle_n = 0
            if turn > 0:
                reward += self.game.make_action([0,0,0,1,0,0])
                self.recording()
                angle_n = self.game.get_game_variable(vizdoom.GameVariable.ANGLE)
                if cross_zero and angle_n<angle_l:
                    cross_zero = False
                if not cross_zero:
                    if angle_n > target:
                        break
            if turn < 0:
                reward += self.game.make_action([0,0,0,0,1,0])
                self.recording()
                angle_n = self.game.get_game_variable(vizdoom.GameVariable.ANGLE)
                if cross_zero and angle_n>angle_l:
                    cross_zero = False
                if not cross_zero:
                    if angle_n < target:
                        break
            if turn == 0:
                return reward, False

            angle_l = angle_n    
            if self.game.is_episode_finished():
                return reward,True
        # print("exit_turn_angle")
        return reward,False
    
    def reward(self,action):
        reward=0
        kill_count = self.game.get_game_variable(vizdoom.GameVariable.KILLCOUNT)
        reward += 2*max(kill_count-self.kill_count,0)
        self.kill_count = kill_count
        return reward
    def _get_player_pos(self):
        """Returns the player X- and Y- coordinates."""
        return self.game.get_game_variable(vizdoom.GameVariable.POSITION_X), self.game.get_game_variable(
                            vizdoom.GameVariable.POSITION_Y)
    def reset(self,seed=None) -> Frame:
        """Resets the environment.

        Returns:
                        The initial state of the new environment.
        """
        # print("[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]")
        self.game.new_episode()
        self.state = self._get_frame()
        self.state = self.empty_frame#{"image": self.empty_frame, "vector": [0,0]}
        self.kill_count = 0
        self.record = []
        self.w +=1
        self.last_x, self.last_y = self._get_player_pos()
        self.health = self.game.get_game_variable(vizdoom.GameVariable.HEALTH)
        self.prompt_gamma = max(self.prompt_gamma - (self.prompt_gamma_o/self.decay_epoch),0)
        return self.state,{}

    def close(self) -> None:
        self.game.close()

    def render(self, mode='human'):
        pass

    def _get_frame(self, done: bool = False) -> Frame:
        return self.frame_processor(self.game.get_state().screen_buffer) if self.game.get_state() and not done else self.empty_frame


class DoomWithBots(DoomEnv):

    def __init__(self, game, frame_processor, frame_skip, n_bots):
        super().__init__(game, frame_processor, frame_skip)
        self.n_bots = n_bots
        self.last_frags = 0
        self._reset_bots()

        # Redefine the action space using combinations.
        self.possible_actions = get_available_actions(np.array(game.get_available_buttons()))
        self.action_space = spaces.Discrete(len(possible_actions))

    def step(self, action):
        self.game.make_action(self.possible_actions[action], self.frame_skip)

        # Compute rewards.
        frags = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        reward = frags - self.last_frags
        self.last_frags = frags

        # Check for episode end.
        self._respawn_if_dead()
        done = self.game.is_episode_finished()
        self.state = self._get_frame(done)

        return self.state, reward, done, {}

    def reset(self,seed = None):
        self._reset_bots()
        self.last_frags = 0

        return super().reset()

    def _respawn_if_dead(self):
        if not self.game.is_episode_finished():
            if self.game.is_player_dead():
                self.game.respawn_player()

    def _reset_bots(self):
                # Make sure you have the bots.cfg file next to the program entry point.
        self.game.send_game_command('removebots')
        for i in range(self.n_bots):
            self.game.send_game_command('addbot')


def create_env(**kwargs) -> DoomEnv:
        # Create a VizDoom instance.
    game = vizdoom.DoomGame()
    game.load_config(f'scenarios/D3_battle_eval.cfg')
    game.set_window_visible(False)
    game.set_death_penalty(2)
    game.set_labels_buffer_enabled(True)
    game.add_available_game_variable(vizdoom.GameVariable.ANGLE)
    game.init()

    # Wrap the game with the Gym adapter.
    return DoomEnv(game, **kwargs)


def make_env(**kwargs) -> Callable:
        def _init() ->  DoomEnv:
            # env = iGibsonEnv(
            #     config_file=os.path.join(igibson.configs_path, config_file),
            #     mode="headless",
            #     action_timestep=1 / 10.0,
            #     physics_timestep=1 / 120.0,
            # )
            game = vizdoom.DoomGame()
            game.load_config(f'github/ViZDoom/scenarios/D3_battle.cfg')
            game.set_window_visible(True)
            game.set_death_penalty(2)
            game.set_labels_buffer_enabled(True)
            game.init()
    # Wrap the game with the Gym adapter.
            return DoomEnv(game, **kwargs)

        return _init

    # Multiprocess
    # env = SubprocVecEnv([make_env(i) for i in range(num_environments)])
    # env = VecMonitor(env)

def create_env_with_bots(scenario, **kwargs) -> DoomEnv:
        # Create a VizDoom instance.
    game = vizdoom.DoomGame()
    game.load_config(f'scenarios/D3_battle.cfg')
    game.add_game_args('-host 1 -deathmatch +viz_nocheat 0 +cl_run 1 +name AGENT +colorset 0' +
                                   '+sv_forcerespawn 1 +sv_respawnprotect 1 +sv_nocrouch 1 +sv_noexit 1')
    game.set_window_visible(False)
    game.init()

    return DoomWithBots(game, **kwargs)


def create_vec_env(n_envs=1, **kwargs) -> vec_env.VecTransposeImage:
    return vec_env.VecTransposeImage(vec_env.SubprocVecEnv([lambda: create_env(**kwargs)] * n_envs))


def vec_env_with_bots(n_envs=1, **kwargs) -> vec_env.VecTransposeImage:
    return vec_env.VecTransposeImage(vec_env.DummyVecEnv([lambda: create_env_with_bots(**kwargs)] * n_envs))


def create_eval_vec_env(**kwargs) -> vec_env.VecTransposeImage:
    return create_vec_env(n_envs=1, **kwargs)

