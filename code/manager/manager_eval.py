import imageio
#from envs import *
#from shooting_env import *
#from john_wei_env import *
#from new_env import *
#from new_shooting_env import *
from manager_env import *
#from manager_env import *
from stable_baselines3 import PPO
import cv2
from tqdm import tqdm, trange
from extractor import CustomCNN,CustomCombinedExtractor
import vizdoom as vzd
def make_gif(agent, file_path):
        def frame_processor(frame):
            frame_processor = lambda frame: cv2.resize(frame[40:, 4:-4], None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
            #print(frame.shape)
            if frame.shape[0] == 3:
                frame = np.moveaxis(frame,0,-1)
            new_frame = frame_processor(frame)
            #print(new_frame.shape)
            return new_frame
        env_args = {
            # 'scenario': 'D3_battle', 
            'frame_skip': 4, 
            'frame_processor': frame_processor,
            'evl':True
        }

        env = create_env(**env_args)
        #env.venv.envs[0].game.set_seed(0)
                
        images = []
        frame_proccessor = lambda frame: cv2.resize(frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        avg_total_reward=0
        num = 20
        gif = 2
        kill_count=0
        for i in trange(num):
            obs ,_= env.reset()

            done = False
            total_reward=0
            while not done:
                action, _ = agent.predict(obs)
                kill = env.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
                obs, reward, done,truncate,info = env.step(action)
                if done:
                    #kill_count += env.venv.envs[0].game.get_available_game_variables()[-1]
                    kill_count+=kill
                    #print(info)
                    if i <gif:
                        images = images+info['info']
                    
                total_reward+=reward
            avg_total_reward+=total_reward/num
            #print(f'total reward for episode: {total_reward}')
        print(images[0].shape)
        print(f'average total reward :{avg_total_reward}')
        print(f'average total reward :{kill_count/num}')
        imageio.mimsave(file_path, images, duration=50)

        env.close()
model = PPO.load('./logs/model/D3_battle/multi/best_model', custom_objects={'policy_kwargs': {'features_extractor_class':CustomCombinedExtractor}})
make_gif(model,'figure/multi.gif')

