import os
import pandas as pd
import numpy as np
from sumTree import SumTree
from PIL import Image


class PER():
    def __init__(self, capacity, screen):
        # Initializing SumTree
        self.capacity = capacity
        self.tree = SumTree(self.capacity)
        # Setting hyperparameters, see README
        self.PER_e = 0.01  # Hyperparameter e 
        self.PER_a = 0.6  # Hyperparameter a 
        self.PER_b = 0.4  # Hyperparameter b
        # Hyperparameter b increment
        self.PER_b_increment_per_sampling = 0.0002
        # Minimum priority
        self.p_min = self.PER_e ** self.PER_a     
        self.clip_priority = self.PER_e ** self.PER_a     
        # Experience indices
        self.experience_ind = 0
        # Screen Surface
        self.screen = screen
        
    
    def add(self, experience):
        '''
            Add a new experience to the sumTree
            Args:
                experience (numpy array): Five Data of the transition step:
                    - current state (numpy array of type object) 
                    - reward
                    - action
                    - next state (numpy array of type object)
                    - done (True or False, end of the episode or loss the game)          

        '''
        # New entries get default priorities
        priority = (0.0 + self.PER_e) ** self.PER_a       

        ######## Save experience in disk

        # Create the experience folder and subfolders                
        os.makedirs(f"./dataset/{self.experience_ind}/current_state", exist_ok=True)
        os.makedirs(f"./dataset/{self.experience_ind}/next_state", exist_ok=True)
        os.makedirs(f"./dataset/{self.experience_ind}/ard", exist_ok=True)
        

        # Save current state images
        #for i, current_cap in enumerate(experience[0]):           
        current_cap = Image.fromarray(experience[0]).convert('L')            
        current_cap.save(f"./dataset/{self.experience_ind}/current_state/c_s.png")          
        
        lidar_current_state = pd.DataFrame(np.array(experience[1]).reshape(1, len(experience[1])))
        lidar_current_state.to_csv((f"./dataset/{self.experience_ind}/current_state/lidar.csv"), index=False, mode='w')

        # Save next state images
        
        next_cap = Image.fromarray(experience[4]).convert('L')                       
        next_cap.save(f"./dataset/{self.experience_ind}/next_state/n_s.png")

        lidar_next_state = pd.DataFrame(np.array(experience[5]).reshape(1, len(experience[5])))
        lidar_next_state.to_csv((f"./dataset/{self.experience_ind}/next_state/lidar.csv"), index=False, mode="w")

        # Save action, reward, done
        ard = pd.DataFrame({"action": [experience[2]], "reward": [experience[3]], "done": [experience[6]]})
        ard.to_csv((f"./dataset/{self.experience_ind}/ard/ard.csv"), index=False)

        ########

        # Adding the priority and experience to the SumTree 
        self.tree.add(priority, self.experience_ind)
        self.experience_ind += 1 
        if self.experience_ind >= self.capacity:
            self.experience_ind=0    
        print("EXP ADDED")    
    
    def sample(self, batch_size):
        '''
            Sample a minibatch of experiences from the SumTree
            Args:
                batch_size (int): Size of the minibatch.
            Returns:                
                minibatch_ISWeights (numpy array of type float32): weights for each sample of the minibatch.
                minibatch_tree_idx (list): indices of the sampled SumTree leaf.
                minibatch (numpy array of type object): minibatch of sampled experiences.

        '''
        #minibatch_current_state = []
        #minibatch_next_state = []        
        minibatch_rewards = []
        minibatch_dones = []
        minibatch_actions = []
        minibatch_lidar_c_state, minibatch_lidar_n_state = np.empty((batch_size, 4)), np.empty((batch_size, 4))

        minibatch_ISWeights, minibatch_current_state, minibatch_next_state = np.empty((batch_size, 1), dtype=np.float32), np.empty((batch_size, self.screen.get_width(), self.screen.get_height())), np.empty((batch_size, self.screen.get_width(), self.screen.get_height()))
        minibatch_tree_idx = []
        # Calculating the priority segment        
        priority_segment = self.tree.total_priority / batch_size
        
        # Here we increasing the PER_b each time we sample a new minibatch
        
       
        # Calculating the max_weight

        p_min = self.p_min
        
        max_weight = (p_min * batch_size) ** (-self.PER_b)        
        
        for i in range(batch_size):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
           
            value = np.random.uniform(a, b)
            
            
            """
            Experience that correspond to each value is retrieved
            """
            data_index, priority, tree_idx = self.tree.get_leaf(value)        
            #P(j)
            sampling_probabilities = priority / self.tree.total_priority
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            minibatch_ISWeights[i, 0] = np.power(batch_size * sampling_probabilities, -self.PER_b)/ max_weight

            minibatch_tree_idx.append(tree_idx)            

            ### Get data
            # Getting current state
            
            minibatch_current_state[i, :, :] = np.array(Image.open(f"./dataset/{data_index}/current_state/c_s.png")) / 255.0                                
            lidar_current_state = np.array(pd.read_csv(f"./dataset/{data_index}/current_state/lidar.csv", header=0))
            minibatch_lidar_c_state[i, :] = lidar_current_state
            # Getting next state
            #for j in range(4):
            minibatch_next_state[i, :, :] = np.array(Image.open(f"./dataset/{data_index}/next_state/n_s.png")) / 255.0
            lidar_next_state = np.array(pd.read_csv(f"./dataset/{data_index}/next_state/lidar.csv", header=0))
            minibatch_lidar_n_state[i, :] = lidar_next_state
                       
            ard = pd.read_csv(f"./dataset/{data_index}/ard/ard.csv", header=0)          
            
            
            
            minibatch_rewards.append(ard['reward'])            
            minibatch_actions.append(ard['action'])
            minibatch_dones.append(ard['done'])
        
        
        return minibatch_ISWeights, minibatch_tree_idx, \
              minibatch_current_state, minibatch_lidar_c_state,\
                  np.array(minibatch_rewards), minibatch_actions, minibatch_next_state, minibatch_lidar_n_state, np.array(minibatch_dones)




        
              
    def update(self, minibatch_tree_idx, abs_errors): 
        '''
            Update the priorities of the sampled experience  
            Args:
                minibatch_tree_idx (list): indices of the sampled SumTree leaf.
                abs_errors ()
            
                
                

        '''       
        new_priorities = (abs_errors + self.PER_e) ** self.PER_a
        
        for tree_idx, new_priority in zip(minibatch_tree_idx, new_priorities):            
            if (new_priority < self.p_min) and (new_priority >= self.clip_priority):
                self.p_min = new_priority
            elif (new_priority < self.clip_priority):
                new_priority = self.clip_priority
            self.tree.update(tree_idx, new_priority)

    def load(self):
        self.tree.load() 
        print("EXPERIENCE LOADED!!!!!")    

    def update_sampling(self):
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
        
     
                   
        
    
