import numpy as np
#import tensorflow as tf
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


######## NETWORK ARCHITECTURE
class duelingDQN(nn.Module):
    def __init__(self, action_dim, name):
        super(duelingDQN, self).__init__()

        self.name = name       

        # 1st Input stream 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 2nd Input Stream
        self.dense1 = nn.Linear(4, 64)
        self.dense2 = nn.Linear(64, 64)

        # V stream
        self.V_dense = nn.Linear(3200, 256)
        self.V = nn.Linear(256, 1)

        # A stream
        self.A_dense = nn.Linear(3200, 256)
        self.A = nn.Linear(256, action_dim)

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.00025)
        self.loss = nn.MSELoss(reduction='mean')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # First input          
        X1 = F.relu(self.conv1(state[0]))
        X1 = F.relu(self.conv2(X1))
        X1 = F.relu(self.conv3(X1))
        
        X1 = X1.view(X1.size(0), -1)
        

        # Second input
        
        X2 = F.relu(self.dense1(state[1]))
        X2 = F.relu(self.dense2(X2))        
        
        X2 = X2.view(X2.size(0), -1)
        
        # Concatenate input streams
        X = torch.cat([X1, X2], dim=1)

        # V stream
        V = F.relu(self.V_dense(X))
        V = self.V(V)

        # A stream
        A = F.relu(self.A_dense(X))
        A = self.A(A)

        return V, A
    
    def save_checkpoint(self, f):
        print("... saving checkpoint ...")
        os.makedirs(f'./records/networks/{f}', exist_ok=True)
        self.chkpt_dir = f"./records/networks/{f}/"
        self.checkpoint_file = os.path.join(self.chkpt_dir, self.name)
        torch.save(self.state_dict(), self.checkpoint_file)

    


class DRL_algorithm:

    def __init__(self, memory, replay_exp_initial_condition):

        self.training_finished = False
        self.update_network_counter = 1
        
        # Epsilon greedy exploration parameters
        self.epsilon = 1.0
        self.epsilon_final = 0.05        
        self.epsilon_interval = self.epsilon - self.epsilon_final
        self.epsilon_greedy_frames = 300_000

        self.memory = memory
        self.replay_exp_initial_condition = replay_exp_initial_condition        
        
        # Batch size
        self.batch_size = 32
        # Number of action of the agent
        self.action_dim = 9
        
        self.discount = 0.99        
        
        self.p_action = 0
        self.same_action_counter = 0

        ####### MODELS        
        # Q network
        self.q_net = duelingDQN(self.action_dim,                                
                                name='duelingDQN')          
        # Q target network
        self.q_target_net = duelingDQN(self.action_dim,                                       
                                       name='target_duelingDQN')              
        # Learning rate decay
        self.q_target_net.load_state_dict(self.q_net.state_dict())

    def policy(self, state, lidar_state):
        if (np.random.uniform(0.0,1.0) < self.epsilon) or (self.memory.storage <= self.replay_exp_initial_condition): 
            # If the random number is less than epsilon
            # Choose a random action
            action = int(np.random.choice(9,1))
            # Counter of repeated action in the previous step
            if self.p_action == action:
                self.same_action_counter += 1
            self.p_action = action
                                            
            return action
        else:
            # Preprocessing state images                        
            img_state = Image.fromarray(state).convert('L') 
            img_state = img_state.rotate(-90)   
            img_state = img_state.transpose(Image.FLIP_LEFT_RIGHT)          
            img_state = img_state.resize((84, 84))                
            img_state = np.array(img_state) / 255.0            
            

            # Expand dimensions             
            img_state_tensor = torch.tensor(img_state).to(self.q_net.device)
            img_state_tensor = img_state_tensor.unsqueeze(0)
            img_state_tensor = img_state_tensor.unsqueeze(0)
            # Lidar state
            lidar_state = np.array(lidar_state) / 200.0
            
            lidar_state_tensor = torch.tensor(lidar_state).to(self.q_net.device)    
            lidar_state_tensor = lidar_state_tensor.unsqueeze(0)
            # Prediction            
            
            V, A = self.q_net.forward([img_state_tensor.float(), lidar_state_tensor.float()])                        
            
            A_mean = torch.mean(A)       
            action = torch.argmax((V + (A - A_mean))).item()            
            
            if self.p_action == action:
                self.same_action_counter += 1
            self.p_action = action

            if self.same_action_counter == 30:                
                action = int(np.random.choice(9,1))                                    
            print("Action taken by the network.!!!!!!!!!!!!!!!!!")
            return action
        
    def train(self):
        # Start training condition
        if self.memory.storage <= self.replay_exp_initial_condition:
            self.training_finished = True
            return            
        # Sampling minibatch                
        states_batch, lidar_c_states, rewards_batch, minibatch_actions, next_states_batch, lidar_n_states, dones_batch = \
            self.memory.sample(self.batch_size)
        
        states = torch.tensor(states_batch).to(self.q_net.device)
        states = states.unsqueeze(1)
        lidar_current_states = torch.tensor(lidar_c_states).to(self.q_net.device)
        next_states = torch.tensor(next_states_batch).to(self.q_net.device)
        next_states = next_states.unsqueeze(1)
        lidar_next_states = torch.tensor(lidar_n_states).to(self.q_net.device)
        rewards = torch.tensor(rewards_batch).to(self.q_net.device)
        actions = torch.tensor(minibatch_actions).to(self.q_net.device)
        dones = torch.tensor(dones_batch).to(self.q_net.device)
        
        self.q_net.optimizer.zero_grad()

        indices = np.arange(self.batch_size)
        
        V_s, A_s = self.q_net.forward([states.float(), lidar_current_states.float()]) 
        V_s_target, A_s_target = self.q_target_net.forward([next_states.float(), lidar_next_states.float()])
        V_s_eval, A_s_eval = self.q_net.forward([next_states.float(), lidar_next_states.float()])
        
        q_pred = torch.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = torch.add(V_s_target, (A_s_target - A_s_target.mean(dim=1, keepdim=True)))                           
        q_eval = torch.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))   

        max_actions = torch.argmax(q_eval, dim=1) 
        
        
        dones_expanded = dones.expand_as(q_next)
        
        q_next[dones_expanded] = 0.0
        
               
        
        q_target = torch.add(rewards.squeeze(), self.discount * q_next[indices, max_actions])        
        loss = self.q_net.loss(q_target.double(), q_pred.double()).to(self.q_net.device)    
        loss.backward()
        self.q_net.optimizer.step()        
        print("Training... ")        
        # Training network       
        # Update epsilon
         # Decay probability of taking random action
        self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
        self.epsilon = max(self.epsilon, self.epsilon_final)
        
        self.update_network_counter += 1        

        if self.update_network_counter == 8000:
            # Copy weights
            self.q_target_net.load_state_dict(self.q_net.state_dict())
            self.update_network_counter = 1

        self.training_finished = True
        print("Training Finished")

    def save_model(self, f):        
        self.q_net.save_checkpoint(f)
        self.q_target_net.save_checkpoint(f)
        print("Models saved")
    
    def load_models(self, f):        
        self.q_net.load_state_dict(torch.load(f"./records/networks/{f}/duelingDQN"))          
        self.q_target_net.load_state_dict(torch.load(f"./records/networks/{f}/target_duelingDQN"))          
        self.q_target_net.load_state_dict(self.q_net.state_dict())  
        print("Models loaded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
