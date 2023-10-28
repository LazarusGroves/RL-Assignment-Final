import gym
import minihack
from nle import nethack
from nle.nethack import Nethack
from minihack import RewardManager
import numpy as np
class EnvGenerator:
    
    # Initilize the environment
    def __init__(self,name): 
        self.name=name
        self.act1 = False # Used to check if the agent has already done a certain action
        self.act2 = False # ""
        self.act3 = False # ""
        self.act4 = False # ""
        self.act5 = False # ""
        self.observation_keys = ("colors", "chars", "glyphs", "pixel", "blstats","message","glyphs_crop") # Mediums of observation
        self.doorPos = None # Used to store the position of the door
        self.countEmpty = -1 # Used to count the number of empty spaces in the environment that have been "seen"
        self.prevPos = None # Used to store the previous position of the agent
        self.rewardWin = 10 # Reward for winning the environment by default

        #   =====================================
        #   MiniHack-Quest-Easy-v0 / MiniHack-Quest-Hard-v0
        #   =====================================
        if (self.name=="MiniHack-Quest-Hard-v0" or self.name=="MiniHack-Quest-Easy-v0"):
            self.rewardWin=100

        #   =====================================
        #   MiniHack-MazeWalk-Mapped-9x9-v0
        #   =====================================
        if(self.name=="MiniHack-MazeWalk-Mapped-9x9-v0"):
            self.rewardWin=50

    # Reset the non repeatable rewards
    def resetBool(self): 
        self.act1 = False
        self.act2 = False
        self.act3 = False  
        self.act4 = False
        self.act5 = False

        self.doorPos = None 
        self.countEmpty = -1
        self.prevPos = None

    # Get the patience for the environment
    def getPatience(self): 
        #  =====================================
        #  MiniHack-Room-5x5-v0
        #  =====================================
        if self.name == "MiniHack-Room-5x5-v0":
            return 9
        
        #  =====================================
        #  MiniHack-Room-15x15-v0
        #  =====================================
        if self.name == "MiniHack-Room-15x15-v0":
            return 9
        
        #  =====================================
        #  MiniHack-MazeWalk-9x9-v0
        #  =====================================
        if self.name == "MiniHack-MazeWalk-9x9-v0":
            return 10000
        
        #  =====================================
        #  MiniHack-LavaCross-Levitate-Potion-Inv-Full-v0
        #  =====================================
        if self.name == "MiniHack-LavaCross-Levitate-Potion-Inv-Full-v0":
            return 15
        
        #  =====================================
        #  MiniHack-KeyRoom-Fixed-S5-v0
        #  =====================================
        if self.name == "MiniHack-KeyRoom-Fixed-S5-v0":
            return 20
        
        #  =====================================
        #  MiniHack-MazeWalk-Mapped-9x9-v0
        #  =====================================
        if self.name=="MiniHack-MazeWalk-Mapped-9x9-v0":
            return 10000  #turn off early stopping
        
        #  =====================================
        #  MiniHack-WoD-Easy-Full-v0
        #  =====================================
        if self.name=="MiniHack-WoD-Easy-Full-v0":
            return 10
        
        #  =====================================
        #  MiniHack-Quest-Easy-v0
        #  =====================================
        if self.name=="MiniHack-Quest-Easy-v0" :
            return 10
        
        #  =====================================
        #  MiniHack-Quest-Hard-v0
        #  =====================================
        if self.name=="MiniHack-Quest-Hard-v0":
            return 100000
    
    # Get the action space for the environment
    def getActionSpace(self): 

        Mov_Space=tuple(nethack.CompassDirection)

        #   =====================================
        #   MiniHack-LavaCross-Levitate-Potion-Inv-Full-v0
        #   =====================================  
        if(self.name == "MiniHack-LavaCross-Levitate-Potion-Inv-Full-v0"):
            Mov_Space += (
                nethack.Command.QUAFF, #drinks potions
                nethack.Command.FIRE, #does the action
            )

        #   =====================================
        #   MiniHack-Quest-Easy-v0
        #   =====================================  
        elif(self.name=="MiniHack-Quest-Easy-v0"):
            Mov_Space+=(
                nethack.Command.ZAP,
                nethack.Command.FIRE,
                nethack.Command.RUSH,
            )

        #   =====================================
        #   MiniHack-Quest-Hard-v0
        #   =====================================     
        elif(self.name=="MiniHack-Quest-Hard-v0"):
            Mov_Space+=(
                nethack.Command.QUAFF,
                nethack.Command.ZAP,
                nethack.Command.FIRE,
                nethack.Command.RUSH,
                nethack.Command.INVOKE,
                nethack.Command.PUTON,
                nethack.Command.WEAR,
            )

        #   =====================================
        #   MiniHack-KeyRoom-Fixed-S5-v0
        #   =====================================    
        elif(self.name == "MiniHack-KeyRoom-Fixed-S5-v0"):
            Mov_Space+=(
                nethack.Command.PICKUP,
                nethack.Command.APPLY, #unlock the door
            )

        #   =====================================
        #   MiniHack-MazeWalk-Mapped-9x9-v0 / MiniHack-MazeWalk-9x9-v0
        #   =====================================    
        elif(self.name=="MiniHack-MazeWalk-Mapped-9x9-v0" or self.name=="MiniHack-MazeWalk-9x9-v0"):
            Mov_Space=(
                nethack.CompassDirection.N,
                nethack.CompassDirection.S,
                nethack.CompassDirection.E,
                nethack.CompassDirection.W
            )

        #   =====================================
        #   MiniHack-WoD-Easy-Full-v0
        #   =====================================
        elif(self.name=="MiniHack-WoD-Easy-Full-v0"):
            Mov_Space+=(
                nethack.Command.ZAP,
                nethack.Command.FIRE
            )
            
        # Return the Movement Space specific to the environment
        return(Mov_Space)
    
    # Get the reward for the environment, based on the message and the characters, this acts as our reward shaping
    def getReward(self,message, chars): 
        # Initially set the reward to 0
        reward = 0
        
        #   =====================================
        #   MiniHack-WoD-Easy-Full-v0
        #   =====================================
        if(self.name=="MiniHack-WoD-Easy-Full-v0"):
            if("wall" in message or "stone" in message):
                reward -= 1
            if("zap" in message):
                reward += 1
            if("direction" in message):
                reward+=0.5

        #   =====================================
        #   MiniHack-Room-5x5-v0
        #   =====================================
        if(self.name=="MiniHack-Room-5x5-v0"):
            if("wall" in message or "stone" in message):
                reward -= 1

        #   =====================================
        #   MiniHack-LavaCross-Levitate-Potion-Inv-Full-v0
        #   =====================================
        if(self.name=="MiniHack-LavaCross-Levitate-Potion-Inv-Full-v0"):
            if("drink" in message and self.act1==False):
                self.act1 = True
                reward += 1
            if("float" in message and self.act2==False):
                self.act2 = True
                reward += 1
            if("wall" in message or "throw" in message):
                reward -= 1

        #   =====================================
        #   MiniHack-MazeWalk-9x9-v0
        #   =====================================
        if(self.name=="MiniHack-MazeWalk-9x9-v0"):
            tempChars=chars-32
            if(self.countEmpty==-1):
                self.countEmpty = np.count_nonzero(tempChars)
            if("wall" in message or "stone" in message):
                reward -= 1
            if(62 in chars and self.act1==False):
                self.act1 = True
                reward += 1
            if(self.countEmpty<np.count_nonzero(tempChars) and 62 not in chars):
                reward += 0.5
                self.countEmpty = np.count_nonzero(tempChars)
            elif(self.countEmpty<np.count_nonzero(tempChars) and 62 in chars):
                reward-=1
                self.countEmpty = np.count_nonzero(tempChars)

        #   =====================================
        #   MiniHack-MazeWalk-Mapped-9x9-v0
        #   =====================================
        if(self.name=="MiniHack-MazeWalk-Mapped-9x9-v0"):
              if("wall" in message or "stone" in message):
                reward -= 1

        #   =====================================
        #   MiniHack-Room-15x15-v0
        #   =====================================
        if(self.name=="MiniHack-Room-15x15-v0"):
            if("wall" in message or "stone" in message):
                reward -= 1
        
        #   =====================================
        #   MiniHack-KeyRoom-Fixed-S5-v0
        #   =====================================
        if(self.name=="MiniHack-KeyRoom-Fixed-S5-v0"):
            reward+=0.09
            if self.doorPos==None and 43 in chars:
                #door pos is the position of the + character
                self.doorPos = np.where(chars==43)
            if('floor' in message or 'nothing' in message):
                reward -= 1
            if("wall" in message or "stone" in message):
                reward -= 1
            if( "see" in message and self.act1==False):
                self.act1 = True
                reward += 0.4
            if( "key" in message and self.act2==False and "see" not in message):
                self.act2 = True
                reward += 1
            # Now check if the door is open by checking if the + is still there
            if(self.doorPos!=None and not chars[self.doorPos]==43 and self.act3==False):
                reward +=5
                print("Door Opened")
                self.act3 = True

        #   =====================================
        #   MiniHack-Quest-Easy-v0
        #   =====================================
        if( self.name == "MiniHack-Quest-Easy-v0"):
            if(self.prevPos == None):
                self.prevPos=np.where(chars==64)[1]
            if ("wall" in message or "stone" in message):
                reward -= 1
            if ("zap" in message and self.act1==False):
                self.act1 = True
                reward += 1
            if ("solidifies" in message and self.act2==False):
                self.act2 = True
                reward += 1
            if("kill" in message):
                reward += 0.5
            if("bites" in message):
                reward -= 1 
            #check if the postiion of 64 has a greater greater column position than any positon of a 125
            if(64 in chars and 125 in chars and np.where(chars==64)[1]<np.where(chars==125)[0][1]):
                reward += 1  
                print("past lava")
        
        #   =====================================
        #   MiniHack-Quest-Hard-v0
        #   =====================================
        if(self.name=="MiniHack-Quest-Hard-v0"):
            #reward for passing through door one, then before passign through the next one, must pick up boots, equip themm, then pass through the next door
            if("wall" in message or "stone" in message):
                reward -= 1
            # Explore Rewards
            tempChars=chars-32
            if(self.countEmpty==-1):
                self.countEmpty = np.count_nonzero(tempChars)
            if("wall" in message or "stone" in message):
                reward -= 1
            if(self.act3==False):
                if(self.countEmpty<np.count_nonzero(tempChars) and 43 not in chars): #used to reward for exploring
                    reward += 0.5
                    self.countEmpty = np.count_nonzero(tempChars)
                elif(self.countEmpty<np.count_nonzero(tempChars) and 43 in chars): #used to stop exploring once door is found
                    reward-=1
                    self.countEmpty = np.count_nonzero(tempChars)
                #if message contains the word "wand" or "boots" then we should give a reward
                if ("door" in message ):
                    reward += 1
                    self.act3 = True
            if("see" in message and self.act4==False ):
                self.act4 = True
                reward+=0.1
            if(("wand" in message or "boots" in message or "horn" in message or "ammulet" in message )and not "see" in message and self.act5==False ):
                reward += 1
                self.act5 = True
            if("kill" in message):
                reward += 10
            if( "nothing to wear" in message):
                reward -= 1
            if("float" in message  and self.act2==False):
                reward += 1
                self.act2 = True
            if ("solidifies" in message and self.act2==False):
                self.act2 = True
                reward += 1
            if("invokes" in message and self.act2==False):
                self.act2 = True
                reward += 0.9
            if("invokes" in message and self.act2==True):
                reward -= 1
            if ("zap" in message and self.act1==False):
                self.act1 = True
                reward += 1
            if("zap" in message and self.act1==True):
                reward -= 0.9
        
        # Return environment specific reward
        return(reward)

    # Create the environment and return it
    def makeEnv(self): 
        return(gym.make(
                        self.name, 
                        actions             =   self.getActionSpace(),
                        observation_keys    =   self.observation_keys, 
                        reward_win          =   self.rewardWin, 
                        reward_lose         =   -5, 
                        penalty_step        =   -0.1, 
                        penalty_mode        =   "always"
                        ))