import gym
import torch
import numpy as np
from model_two.PPO import PPO
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import pygame as pg
from tqdm import trange, tqdm
device="cuda"


class Agent:

    def __init__(self,env, hyper_params,env_gen): #env_gen is the environment generator
        self.env=env
        self.hyper_params=hyper_params
        self.obs_dim=hyper_params["obs_dim"]
        self.action_space=self.env.action_space.n
        self.actor=PPO(self.obs_dim, self.action_space).to(device)
        self.critic=PPO(self.obs_dim, 1).to(device)
        self.actor_optim=Adam(self.actor.parameters(), lr=hyper_params["lr"])
        self.critic_optim=Adam(self.critic.parameters(), lr=hyper_params["lr"])
        self.cov_var = torch.full(size=(self.action_space,), fill_value=0.5).to(device)
        self.cov_mat = torch.diag(self.cov_var).to(device)
        self.best_actor=None
        self.best_critic=None
        self.best_returns=-10000
        self.patience=0
        self.returns=[]
        self.env_gen=env_gen
        np.random.seed(hyper_params["seed"])
        torch.manual_seed(hyper_params["seed"])
        if(hyper_params["visualise"]):
            pg.init()
            pg.font.init()
            self.screen=pg.display.set_mode((900,900))
    def formatState(self,state): #state is a numpy array
        state=np.array(state)/np.max(state) #normalise
        #reorder state so that its in the correct order
        state=state.reshape((1,1,21,79))
        #convert to tensor
        state=torch.from_numpy(state).float().squeeze(0).to(device) #squeeze to remove the first dimension
        state=torch.unsqueeze(state,0) #add the batch dimension
        return(state) #return the state as a tensor

    def learn(self,total_time_steps): #total_time_steps is the total number of time steps to train for
        time_steps_so_far = 0  # how many time steps we have trained for so far
        iterations = 0 #how many iterations we have done so far
        
        while(time_steps_so_far<total_time_steps): #while we have not trained for the total number of time steps
            print("time steps so far: "+ str(time_steps_so_far)+ "/"+ str( total_time_steps), end="\r") #used a progress for how far into training we are
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.replay_buffer() #get the batch of data
            time_steps_so_far += np.sum(batch_lens) #add the number of time steps in the batch to the total number of time steps
            #average the batch_rtgs for best returns
            curr_mean=np.mean(batch_rtgs.detach().cpu().numpy())

            if(curr_mean>self.best_returns): #if the current mean is better than the best returns so far
                self.best_returns=curr_mean #update the best returns
                self.best_actor=self.actor.state_dict() #update the best actor
                self.best_critic=self.critic.state_dict() #update the best critic
                self.patience = 0 #reset the patience
            elif iterations>10: #if we have done more than 10 iterations
                self.patience += 1 #increase the patience

            if(self.patience>self.hyper_params["patience"]): #if we have reached the patience limit
                print("🛑Early Stopping") 
                self.actor.load_state_dict(self.best_actor) #load the best actor
                self.critic.load_state_dict(self.best_critic) #load the best critic
                break #stop training

            iterations += 1 #increase the number of iterations
            
            V,_ = self.evaluate(batch_obs, batch_acts) #evaluate the value function
            K_Advtange = batch_rtgs-V.detach() #calculate the advantage function
            
            K_Advtange = (K_Advtange-K_Advtange.mean())/(K_Advtange.std()+1e-10) #normalise the advantage function

            for x in range (self.hyper_params["updates"]): #loop through number of back props we want to do
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts) #evaluate the value function and log probs
                ratio =             torch.exp(curr_log_probs - batch_log_probs) #calculate the ratio
                surrogate1 =        ratio * K_Advtange #calculate the surrogate
                surrogate2 =        torch.clamp(ratio, 1 - self.hyper_params["eps_clip"], 1 + self.hyper_params["eps_clip"]) * K_Advtange #calculate the surrogate
                actor_loss =        (-torch.min(surrogate1, surrogate2)).mean() #calculate the actor loss
                critic_loss =       nn.MSELoss()(V, batch_rtgs) #calculate the critic loss

                self.actor_optim.zero_grad() #zero the gradient
                actor_loss.backward() #backpropagate
                self.actor_optim.step()  #update the actor
                self.critic_optim.zero_grad() #zero the gradient
                critic_loss.backward() #backpropagate
                self.critic_optim.step() #update the critic

            if iterations % self.hyper_params["save_freq"] == 0: #if we have reached the save frequency
                torch.save(self.actor.state_dict(), self.hyper_params["save_path"]+"actor_"+str(iterations)+".pth") #save the actor
                torch.save(self.critic.state_dict(), self.hyper_params["save_path"]+"critic_"+str(iterations)+".pth") #save the critic
                np.save( self.hyper_params["save_path"]+"returns_"+str(iterations)+".npy",self.returns)       #save the returns

        torch.save(self.actor.state_dict(), self.hyper_params["save_path"]+"actor_"+str(iterations)+".pth") #save the actor
        torch.save(self.critic.state_dict(), self.hyper_params["save_path"]+"critic_"+str(iterations)+".pth") #save the critic
        np.save( self.hyper_params["save_path"]+"returns_"+str(iterations)+".npy",self.returns)      #save the returns   
        
    def replay_buffer(self): #sample a batch from the replay buffer
        batch_obs = [] 
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        rewards_per_rollout = []
        t = 0

        self.returns.append(0)
        while t<self.hyper_params["batch_size"]: #while we have not reached the batch size
            self.env_gen.resetBool() #reset the bool
            rewards_per_rollout = [] #reset the rewards per rollout
            obs_full = self.env.reset() #reset the environment
            obs = obs_full["glyphs"] #get the glyphs
            obs = self.formatState(obs) #format the state
            done = False #set done to false
            for t_step in range(self.hyper_params["max_timesteps"]): #loop through the max number of time steps
                t += 1 #increase the number of time steps
                batch_obs.append(obs.detach().cpu().numpy()) #add the obs to the batch
                action, log_prob = self.act(obs) #get the action and log prob
                full_obs, reward, done, info=self.env.step(np.argmax(action)) #take the action
                
                obs = full_obs["glyphs"] #get the glyphs
                obs = self.formatState(obs) #format the state
                chars=full_obs["chars"] #get the chars
                msg = full_obs["message"] #get the message
                
                msg = msg[: np.where(msg == 0)[0][0]].tobytes().decode("utf-8") #convert the message to a string
                if reward == -0.1: #if the reward is -0.1
                    reward += self.env_gen.getReward(msg,chars) #update the reward with custom reward manager
                    
                self.returns[-1] += reward*self.hyper_params["gamma"] #update the returns
                if(self.hyper_params["visualise"]): #if we want to visualise
                    img = full_obs["pixel"] #get the pixels
                    rotated_array = np.rot90(img, k=1)  #rotate the pixels
                    my_surface = pg.pixelcopy.make_surface(rotated_array) #make the surface
                    text_surface = pg.font.Font(None, 36).render(msg, True, (255,255,255)) #render the text
                    text_position = (900// 2 - text_surface.get_width() // 2, 900 - text_surface.get_height() - 20)#get the text position
                    self.screen.fill((0, 0, 0))  #fill the screen
                    self.screen.blit(my_surface,(0,0)) #blit the surface
                    self.screen.blit(text_surface, text_position)# blit the text
                    pg.display.update() #update the display
                
                
                
                batch_acts.append(action) #add the action to the batch
                batch_log_probs.append(log_prob) #  add the log prob to the batch
                rewards_per_rollout.append(reward) #add the reward to the rewards per rollout
                if done: #if done
                    self.returns.append(0) #add a new return
                    break #break
            batch_lens.append(t_step+1) #add the number of time steps to the batch lens
            batch_rews.append(rewards_per_rollout) #add the rewards per rollout to the batch rews
        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(device) #convert to tensor
        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(device) #convert to tensor
        batch_log_probs=torch.tensor(batch_log_probs, dtype=torch.float).to(device) #convert to tensor
        batch_rtgs = self.compute_rtgs(batch_rews) #compute the returns
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens #return the batch
    
    def compute_rtgs(self, batch_rews): #compute the returns
        batch_rtgs = [] #the returns to go
        for episode in reversed(batch_rews): #loop through the batch
            discounted_reward = 0 #the discounted reward
            for reward in reversed(episode): #loop through the episode
                discounted_reward = reward + discounted_reward * self.hyper_params["gamma"] #update the discounted reward
                batch_rtgs.insert(0, discounted_reward) #insert the discounted reward
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(device) #convert to tensor
        return batch_rtgs #return the batch rtgs
    
    def act(self,obs): #get the action
        mean=self.actor(obs).squeeze() #get the mean of the normal distribution centered around the actor output
        dist=MultivariateNormal(mean, self.cov_mat) #create the distribution
        action=dist.sample() #sample from the distribution
        log_prob=dist.log_prob(action) #get the log prob
        return(action.detach().cpu().numpy(), log_prob.detach().cpu()) #return the action and log prob
    
    def evaluate(self,batch_obs,batch_acts): #evaluate the value function and log probs
        batch_obs=batch_obs.reshape((batch_obs.shape[0],1,21,79)) #reshape the batch obs
        V=self.critic(batch_obs).squeeze() #get the value function
        mean=self.actor(batch_obs) #get the mean of the normal distribution centered around the actor output
        dist=MultivariateNormal(mean, self.cov_mat) #create the distribution
        log_probs=dist.log_prob(batch_acts) #get the log probs
        return(V, log_probs) #return the value function and log probs