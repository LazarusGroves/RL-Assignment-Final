import numpy as np
class ReplayBuffer:
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, size):
        """
        Initialise a buffer of a given size for storing transitions
        :param size: the maximum number of transitions that can be stored
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.maxRange=0
    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, next_state, done,td_error):
        """
        Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        :param state: the agent's initial state
        :param action: the action taken by the agent
        :param reward: the reward the agent received
        :param next_state: the subsequent state
        :param done: whether the episode terminated
        """
        
        numTickets      =   1000 # Lottery system tickets
        numTickets      -=  int(td_error*1000) # Lose tickets if the TD error is high
        data            =   (state, action, reward, next_state, done,self.maxRange,self.maxRange+numTickets) 
        self.maxRange   +=  numTickets + 1 # Update the max range

        # If the buffer is full, overwrite the oldest transitions
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    # Enconding sample defined by the lottery system
    def _encode_sample(self, indices):

        states, actions, rewards, next_states, dones,mins,maxs = [], [], [], [], [],[],[]

        for i in indices:
            data = self._storage[i] # Get the data from the lottery system
            state, action, reward, next_state, done,min,max = data 
            states.append(np.array(state, copy=False)) # Append State
            rewards.append(reward) # Append Reward
            next_states.append(np.array(next_state, copy=False)) # Append Next State
            dones.append(done) # Append Done
            mins.append(min) # Append Min 
            maxs.append(max) # Append Max
        
        # Return the rewards, states, next states, dones
        return (
            np.array(states), 
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    # Sample a batch of experiences
    def sample(self, batch_size):
        """
        Use Lottery system to sample a batch of experiences.
        """
        indices=np.zeros(batch_size,dtype=int)
        for i in range(batch_size):
            selected=np.random.randint(0,self.maxRange)
            for j in range(len(self._storage)):
                if(self._storage[j][5]<=selected and self._storage[j][6]>=selected):
                    indices[i]=j
                    break
        return self._encode_sample(indices)