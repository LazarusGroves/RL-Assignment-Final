import numpy as np
import torch
# import tensorflow as tf
import gym
import minihack
import matplotlib.pyplot as plt
from nle import nethack
import pygame as pg
from tqdm import trange
import copy
import random
from model_one.agent import DQNAgent
from model_one.replay_buffer import ReplayBuffer
from pygame.locals import *
from minihack import RewardManager
from environments import EnvGenerator
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import trange
hyper_parameters = {
    "seed": 102,  # which seed to use
    "env": "MiniHack-Quest-Hard-v0",  # name of the game
    "replay-buffer-size": int(5e6),  # replay buffer size
    "learning-rate": 1e-3,  # learning rate for optimizer
    "discount-factor": 0.99,  # discount factor
    # total number of steps to run the environment for
    "num-epochs": 10000, #int(1e6)
    "batch-size": 32,  # number of transitions to optimize at the same time
    "learning-starts": 500,  # number of steps before learning starts used to be 10000
    "learning-freq": 1,  # number of iterations between every optimization step
    "use-double-dqn": True,  # use double deep Q-learning
    "target-update-freq": 100,  # number of iterations between every target network update
    "eps-start": 1.0,  # e-greedy start threshold
    "eps-end": 0.1,  # e-greedy end threshold
    "eps-fraction": 0.7,  # fraction of num-steps
    "print-freq": 10,
    "save_freq": 999,  
    "reward-shaping": True,
    "save_path":   None,
    "visualise": False,
    "record":True
}
hyper_parameters["save_path"]   =   "../save/"+hyper_parameters["env"]+"/DQN_models/"

if(hyper_parameters["visualise"]): #set up pygame
    x=900
    y=900
    pg.init()
    pg.font.init()
    screen=pg.display.set_mode((x,y))
    font=pg.font.Font(None, 20)
device="cuda"

def formatState(state): #format the state to something that can be used by the network
        state=np.array(state)/np.max(state)
        #convert to tensor
        state=state.reshape((1,1,21,79))
        state=torch.from_numpy(state).float().squeeze(0).to(device)
        return(state)



if not os.path.exists("../save/"+hyper_parameters["env"]): # If does not exist make a folder in /save with the name of the environment
    os.mkdir("../save/"+hyper_parameters["env"]) # Make a folder in /save/env/DQN_models

# Make a folder in /save/env/PPO_models
if not os.path.exists("../save/"+hyper_parameters["env"]+"/DQN_models"):
    os.mkdir("../save/"+hyper_parameters["env"]+"/DQN_models")
envGen=EnvGenerator(hyper_parameters["env"]) #get the environment generator
env=envGen.makeEnv() #make the environment



env.seed(hyper_parameters["seed"]) #set the seed
np.random.seed(hyper_parameters["seed"]) #set the seed
random.seed(hyper_parameters["seed"])

state = env.reset()['glyphs']# Get the initial state

# Print glyph names and corresponding indices in a list
state=formatState(state) #format the state
obs_space_size = 1 #env.observation_space.shape[0]
action_space_size = env.action_space.n #env.action_space.shape[0]


replay_buffer = ReplayBuffer(hyper_parameters["replay-buffer-size"]) #create the replay buffer
agent = DQNAgent(obs_space_size, action_space_size, replay_buffer, True, #create the agent
                 hyper_parameters["learning-rate"], hyper_parameters["batch-size"], hyper_parameters["discount-factor"])

eps_timesteps = hyper_parameters["eps-fraction"] * \
    float(hyper_parameters["num-epochs"]) #calculate the number of steps to decay epsilon over
episode_rewards = [0.0] #initialize the episode rewards

for t in trange(hyper_parameters["num-epochs"]): #iterate over the number of epochs
    fraction = min(1.0, float(t) / eps_timesteps) #calculate the fraction of the way through epsilon decay we are
    eps_threshold = hyper_parameters["eps-start"] + fraction * (
        hyper_parameters["eps-end"] - hyper_parameters["eps-start"]
    )
    sample = random.random() #sample a random number
    if (sample < eps_threshold): #if the sample is less than epsilon then explore
        action = env.action_space.sample() #sample a random action
    else:
        action = agent.act(state) #otherwise act greedily
    
    next_state_all, reward, done, info = env.step(action) #take a step in the environment
    
   
    msg=next_state_all['message'] #get the message
    msg = msg[: np.where(msg == 0)[0][0]].tobytes().decode("utf-8")#convert the message to a string
   
     
    reward+=envGen.getReward(msg,next_state_all['chars']) #Adjust the rewards with the custom reward manager
    if(hyper_parameters["visualise"]): #if we are visualising then render the environment
        img=rotated_array = np.rot90(next_state_all['pixel'], k=-1) #rotate the image
        my_surface = pg.pixelcopy.make_surface(img) #make a surface from the image
        text_surface = font.render(msg, True, (255,255,255)) #render the message
        text_position = (900// 2 - text_surface.get_width() // 2, 900 - text_surface.get_height() - 20) #get the position of the message
        screen.fill((0, 0, 0)) #fill the screen
        screen.blit(my_surface,(0,0)) #blit the image
        screen.blit(text_surface, text_position) #blit the message
        pg.display.update() #update the display
    next_state = next_state_all['glyphs'] #get the next state
    next_state=formatState(next_state) #format the next state
    copyState=copy.deepcopy(state) #copy the state
    copyState2=copy.deepcopy(state) #copy the state
    copyNextState=copy.deepcopy(next_state) #copy the next state
    copyNextState2=copy.deepcopy(next_state) #copy the next state
    copyState=copyState.cpu().detach().numpy() #convert to numpy
    copyNextState=copyNextState.cpu().detach().numpy() #convert to numpy
    num_episodes = len(episode_rewards) #get the number of episodes
    
    copyState2=torch.unsqueeze(copyState2,0) #unsqueeze the state
    copyNextState2=torch.unsqueeze(copyNextState2,0) #unsqueeze the next state
    Qi=np.max(agent.policy_network(copyNextState2).detach().cpu().numpy()) #get the max q value of the next state
    Q=np.max(agent.policy_network(copyState2).detach().cpu().numpy()) #get the max q value of the state
    
    tdError=reward+hyper_parameters["discount-factor"]*Qi-Q #calculate the td error
    agent.replay_buffer.add(copyState, action, reward, copyNextState, done,tdError) #add the transition to the replay buffer
    episode_rewards[-1] += reward #add the reward to the episode rewards
    state = next_state #set the state to the next state
    if done: #if the episode is done
        state = env.reset()['glyphs'] #reset the environment
        state=formatState(state) #format the state
        episode_rewards.append(0) #append a 0 to the episode rewards

    if (
        len(episode_rewards) % hyper_parameters["print-freq"] == 0 #print the episode rewards
        and t > hyper_parameters["learning-starts"] #if we are past the learning starts
        
    ):
       
        print("********************************************************")
        print("steps: {}".format(t))
        print("episode: {}".format(num_episodes))
        print("episode reward: {}".format(episode_rewards[-2]))
        print("Percent exploring {}".format(eps_threshold*100))
        print("********************************************************")
    if (t> hyper_parameters["learning-starts"] and t % hyper_parameters["learning-freq"] == 0):
        agent.optimise_td_loss() #optimise the td loss
    if (t> hyper_parameters["learning-starts"] and t % hyper_parameters["target-update-freq"] == 0):
        agent.update_target_network() #update the target network
    if (t> hyper_parameters["learning-starts"] and t % hyper_parameters["save_freq"] == 0):
        agent.saveAgent(hyper_parameters["save_path"] ) #save the agent
        np.save(hyper_parameters["save_path"]+"returns.npy",episode_rewards)
        
np.save(hyper_parameters["save_path"]+"returns.npy",episode_rewards) #save the episode rewards
agent.saveAgent(hyper_parameters["save_path"]) #save the agent
plt.plot(episode_rewards) #plot the episode rewards
plt.show() #show the plot




def scale_observation(observation, new_size): 
    """
    Scale an observation (image) to a new size using Pygame.

    Args:
        observation (pygame.Surface): The input Pygame observation.
        new_size (tuple): The new size (width, height) for scaling.

    Returns:
        pygame.Surface: The scaled observation.
    """
    return pg.transform.scale(observation, new_size)
    
def render(obs, screen, font, text_color):
    """
    Render the game observation on the Pygame screen.

    Args:
        obs (dict): Observation dictionary containing "pixel" and "message" keys.
        screen (pygame.Surface): The Pygame screen to render on.
        font (pygame.Font): The Pygame font for rendering text.
        text_color (tuple): The color for rendering text.
    """
    img = obs["pixel"]
    msg = obs["message"]
    msg = msg[: np.where(msg == 0)[0][0]].tobytes().decode("utf-8")
    rotated_array = np.rot90(img, k=1)

    window_size = screen.get_size()
    image_surface = pg.surfarray.make_surface(rotated_array)
    image_surface = scale_observation(image_surface, window_size)

    screen.fill((0, 0, 0))
    screen.blit(image_surface, (0, 0))

    text_surface = font.render(msg, True, text_color)
    text_position = (window_size[0] // 2 - text_surface.get_width() // 2, window_size[1] - text_surface.get_height() - 20)
    screen.blit(text_surface, text_position)
    pg.display.flip()


def record(env,frameRate):
    
    
    frame_width = env.observation_space["pixel"].shape[1]
    frame_height = env.observation_space["pixel"].shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("../save/"+hyper_parameters["env"]+"/DQN_models/vid.mp4", fourcc, frameRate, (frame_width, frame_height))
    pg.init()
    screen = pg.display.set_mode((frame_width, frame_height))
    font = pg.font.Font(None, 36)
    text_color = (255, 255, 255)
    done = False
    obs = env.reset()
    timeStep=0
    while(not done and timeStep<1000):
        state=formatState(obs["glyphs"])
        action=agent.act(state)
        obs, _, done, _ = env.step(action)
        render(obs, screen, font, text_color)
        pg.image.save(screen, "temp_frame.png")
        frame = cv2.imread("temp_frame.png")
        out.write(frame)
        timeStep += 1
    out.release()
    cv2.destroyAllWindows()
    os.remove("temp_frame.png")
if(hyper_parameters["record"]):   
    record(env,10)