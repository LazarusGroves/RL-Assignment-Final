from torch import nn
import numpy as np
class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """
    def __init__(self, observation_space_size, action_space_size):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()

        '''
        self.conv1= nn.Conv2d(1, out_channels=18, kernel_size=3, stride=1) #first conv layer
        self.conv2= nn.Conv2d(18, out_channels=36, kernel_size=4, stride=2) #second conv layer
        self.conv3=nn.Conv2d(36, out_channels=36, kernel_size=2, stride=1) #third conv layer

        self.relu=nn.ReLU() #activation function
        
        linear_input = 36*252#input to linear layer, has to be calculated from the conv layers
        self.linear1=nn.Linear(linear_input,9) #first linear layer
        self.linear2=nn.Linear(9,action_space_size) #second linear layer
        '''
        self.conv1= nn.Conv2d(1, out_channels=20, kernel_size=(5,5))
        self.relu=nn.ReLU()
        self.MaxPool1=nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2= nn.Conv2d(20, out_channels=50, kernel_size=(5,5))

        self.fc1=nn.Linear(1600, 500)
        self.fc2=nn.Linear(500, action_space_size)
        
    def forward(self, x):
        x=self.relu(self.conv1(x)) 
        x=self.MaxPool1(x)
        x=self.relu(self.conv2(x))
        x=self.MaxPool1(x)
        x=x.reshape(x.shape[0], -1)
        x=self.relu(self.fc1(x))
        x=self.fc2(x)

        return x