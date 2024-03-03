import os
import pandas as pd
import numpy as np
from PIL import Image


class ReplayBuffer:

    def __init__(self, max_size, pointer, screen):
        # Memory
        self.storage = 0
        # Memory size
        self.max_size = max_size
        # Puntero a las diferentes celdas en la memoria
        self.experience_ind = pointer        
        self.screen = screen
        self.flag = False

        self.ACTIONS = {
            "NO ACTION": 0,            
            "UP": 1,
            "DOWN": 2,
            "LEFT": 3,
            "RIGHT": 4,
            "DOUBLE-UP": 5,
            "DOUBLE-DOWN": 6,
            "DOUBLE-LEFT": 7,
            "DOUBLE-RIGHT": 8
        }

    def add(self, experience):
        ######## Save experience in disk

        # Create the experience folder and subfolders                
        os.makedirs(f"./dataset/{self.experience_ind}/current_state", exist_ok=True)
        os.makedirs(f"./dataset/{self.experience_ind}/next_state", exist_ok=True)
        os.makedirs(f"./dataset/{self.experience_ind}/ard", exist_ok=True)
        

        # Save current state images
        #for i, current_cap in enumerate(experience[0]):                   

        current_cap = Image.fromarray(experience[0]).convert('L')                    
        current_cap = current_cap.rotate(-90)
        current_cap = current_cap.transpose(Image.FLIP_LEFT_RIGHT)  
        current_cap = current_cap.resize((84,84))                     
        current_cap.save(f"./dataset/{self.experience_ind}/current_state/c_s.png")          
        
        lidar_current_state = pd.DataFrame(np.array(experience[1]).reshape(1, len(experience[1])) / 200.0)        
        lidar_current_state.to_csv((f"./dataset/{self.experience_ind}/current_state/lidar.csv"), index=False, mode='w')

        # Save next state images
        
        next_cap = Image.fromarray(experience[4]).convert('L')                       
        next_cap = next_cap.rotate(-90)
        next_cap = next_cap.transpose(Image.FLIP_LEFT_RIGHT)        
        next_cap = next_cap.resize((84,84))
        next_cap.save(f"./dataset/{self.experience_ind}/next_state/n_s.png")

        lidar_next_state = pd.DataFrame(np.array(experience[5]).reshape(1, len(experience[5])) / 200.0)
        lidar_next_state.to_csv((f"./dataset/{self.experience_ind}/next_state/lidar.csv"), index=False, mode="w")

        # Save action, reward, done
        ard = pd.DataFrame({"action": [experience[2]], "reward": [experience[3]], "done": [experience[6]]})
        ard.to_csv((f"./dataset/{self.experience_ind}/ard/ard.csv"), index=False)

        ########
        
        self.storage = self.experience_ind
        self.experience_ind += 1 
        if self.experience_ind >= self.max_size:
            self.experience_ind = 0    
            self.flag=True
        if self.flag:
            self.storage = self.max_size

        print("EXP ADDED")  
        return

    def sample(self, batch_size):
        # Selecting the experience in memory
        batch = np.random.choice(self.storage, batch_size, replace=False)
        minibatch_rewards = []
        minibatch_dones = []
        minibatch_actions = []
        minibatch_lidar_c_state, minibatch_lidar_n_state = np.empty((batch_size, 4), dtype=float), np.empty((batch_size, 4), dtype=float)
        minibatch_current_state, minibatch_next_state = np.empty((batch_size, 84, 84), dtype=float), np.empty((batch_size, 84, 84), dtype=float)

        for i, data_index in enumerate(batch):
            ### Get data
            # Getting current state
            try:            
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
                
                minibatch_actions.append(self.ACTIONS[ard['action'].values[0]])
                minibatch_dones.append(ard['done'])
                
            except Exception as e:
                print(e)
                
                

        return minibatch_current_state, minibatch_lidar_c_state,\
                  np.array(minibatch_rewards), np.array(minibatch_actions), minibatch_next_state, minibatch_lidar_n_state, np.array(minibatch_dones)