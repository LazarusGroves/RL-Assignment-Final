from model_one.replay_buffer import ReplayBuffer
import torch
import numpy as np
from model_one.DQN import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Most of our code will be hardcoded to assume we are running on the GPU Note

class DQNAgent:
    def __init__(
        self,
        input_size,
        action_space,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
    ):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param input_size: the input size of the state
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """
        self.replay_buffer=replay_buffer #replay buffer
        self.batch_size=batch_size #batch size
        self.double_dqn=use_double_dqn #double dqn
        self.gamma=gamma #discount factor

        self.policy_network=DQN(input_size, action_space).to(device) #policy network
        self.target_network=DQN(input_size, action_space).to(device) #target network
        self.update_target_network() #update the target network to match the policy network
        self.target_network.eval() #set the target network to evaluation mode   

        self.optimizer=torch.optim.Adam(self.policy_network.parameters(), lr=lr) #optimiser
        self.critereon=torch.nn.SmoothL1Loss() #loss function
  
    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        states, actions, rewards, next, dones=self.replay_buffer.sample(self.batch_size) #sample a batch from the replay buffer
        # Normalise the states
        states  =   np.array(states)/np.max(states)
        next    =   np.array(next)/np.max(states)

        # Convert all vars to tensors and move to device
        states  =   torch.from_numpy(states).float().to(device)
        actions =   torch.from_numpy(actions).long().to(device)
        rewards =   torch.from_numpy(rewards).float().to(device)
        next    =   torch.from_numpy(next).float().to(device) 
        dones   =   torch.from_numpy(dones).float().to(device)  # using done (as a float) instead of if statement

        # Zero the gradient to prevent accumulation
        with torch.no_grad(): 
            if self.double_dqn: #if using double dqn
                _, best_action=self.policy_network(next).max(1) #get the best action from the policy network
                q_values=self.target_network(next).gather(1, best_action.unsqueeze(1)).squeeze() #get the q value of the best action from the target network
            else:
                q_values=self.target_network(next).max(1)[0] #get the q value of the best action from the target network
            
            #calculate the target
            target=rewards + (1-dones)*self.gamma*q_values
        
        # Calculate the loss
        inputs=self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze() 
        loss=self.critereon(inputs, target)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      
        # Return loss
        return(loss.item())
        
    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        # Normalise state
        state=torch.unsqueeze(state,0) 
        with torch.no_grad():
            QVal=self.policy_network(state) #get the q values from the policy network
            _,action=QVal.max(1) #get the best action
            return action.item() #return the action
        
    def saveAgent(self,save_path):
        """
        Save the agent's policy network and target network
        """
        torch.save(self.policy_network.state_dict(), save_path+"policy.pth")
        torch.save(self.target_network.state_dict(), save_path+"target.pth")
        print("=== Saved Checkpoint ===")

