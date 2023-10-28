from environments import EnvGenerator
from model_two.agent import Agent
import matplotlib.pyplot as plt
import os

hyper_parameters = {
    "env": "MiniHack-Quest-Hard-v0",  # name of the game
    "seed": 42,                 # which seed to use
    "obs_dim":1,                # observation space dimension
    "lr":0.001,                 # learning rate
    "gamma": 0.95,              # discount factor
    "max_timesteps" : 50,       # total number of iterations/timesteps
    "eps_clip": 0.05,           # clip parameter for PPO
    "updates":1,                # how many times to update the critic per step
    "batch_size" : 100,         # batch size for PPO
    "save_freq": 999,           # how often to save in number of iterations
    "save_path": "PPO_models",  # where to save the actor
    "total_time_steps": 100000,  # total number of time steps to train for
    "patience": 9,              # for 5x5 room
    "visualise": False,
}

# If does not exist make a folder in /save with the name of the environment
if not os.path.exists("../save/"+hyper_parameters["env"]):
    os.mkdir("../save/"+hyper_parameters["env"])

# Make a folder in /save/env/PPO_models
if not os.path.exists("../save/"+hyper_parameters["env"]+"/PPO_models"):
    os.mkdir("../save/"+hyper_parameters["env"]+"/PPO_models")
    
hyper_parameters["save_path"]   =   "../save/"+hyper_parameters["env"]+"/PPO_models/" 
env_gen                         =   EnvGenerator(hyper_parameters["env"])
hyper_parameters["patience"]    =   env_gen.getPatience()
env                             =   env_gen.makeEnv()
agent                           =   Agent(env, hyper_parameters,env_gen)

agent.learn(hyper_parameters["total_time_steps"]) # Learn for total_time_steps

plt.plot(agent.returns) # Plot the returns
plt.show()
