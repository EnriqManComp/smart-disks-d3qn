import numpy as np
#import tensorflow as tf
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


######## NETWORK ARCHITECTURE
'''
class duelingDQN(tf.keras.Model):
    def __init__(self,action_dim):
        super(duelingDQN, self).__init__()
        # 1st Input stream        
        self.conv1 = tf.keras.layers.Conv2D(filters=32,kernel_size=(8,8),strides=4, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=2, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=1, activation='relu')
        self.flatten = tf.keras.layers.Flatten()

        # 2nd Input stream
        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')  
        self.dropout = tf.keras.layers.Dropout(0.2)  
        self.dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        
        # Concatenate layer
        self.concat = tf.keras.layers.Concatenate(axis=1)

        # V stream
        self.V_dense = tf.keras.layers.Dense(units=256, activation='relu')        
        self.V = tf.keras.layers.Dense(1, activation=None)

        # A stream        
        self.A_dense = tf.keras.layers.Dense(units=256, activation='relu')
        self.A = tf.keras.layers.Dense(action_dim, activation=None)
        
    def call(self, state):        
        # First input        
        X1 = self.conv1(tf.expand_dims(state[0], axis=-1))
        X1 = self.conv2(X1)
        X1 = self.conv3(X1)
        X1 = self.flatten(X1)

        # Second input
        #X2 = self.dense1(state[1])        
        X2 = self.dense1(state[1])
        X2 = self.dropout(X2)
        X2 = self.dense2(X2)        
        # Concatenate input streams
        X = self.concat([X1, X2])        

        # V stream
        V = self.V_dense(X)
        V = self.V(V)

        # A stream
        A = self.A_dense(X)
        A = self.A(A)

        # Compute Q value
        Q = (V - ( A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        return Q

    def advantage(self, state):
        # First input        
        X1 = self.conv1(tf.expand_dims(state[0], axis=-1))
        X1 = self.conv2(X1)
        X1 = self.conv3(X1)
        X1 = self.flatten(X1)

        # Second input
        #X2 = self.dense1(state[1])
        X2 = self.dense1(state[1])
        X2 = self.dropout(X2)
        X2 = self.dense2(X2)

        # Concatenate input streams
        
        X = self.concat([X1, X2])

        # A stream
        A = self.A_dense(X)
        A = self.A(A)

        return A 
''' 
class duelingDQN(nn.Module):
    def __init__(self, action_dim, chkpt_dir, name):
        super(duelingDQN, self).__init__()
        
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, name)

        # 1st Input stream 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 2nd Input Stream
        self.dense1 = nn.Linear(4, 64)
        self.dense2 = nn.Linear(64, 64)

        # V stream
        self.V_dense = nn.Linear(256, 256)
        self.V = nn.Linear(256, 1)

        # A stream
        self.A_dense = nn.Linear(256, 256)
        self.A = nn.Linear(256, action_dim)

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.00025)
        self.loss = nn.MSELoss(reduction='none')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # First input
        X1 = F.relu(self.conv1(state[0]))
        X1 = F.relu(self.conv2(X1))
        X1 = F.relu(self.conv3(X1))
        X1 = torch.flatten(X1)

        # Second input
        X2 = F.relu(self.dense1(state[1]))
        X2 = F.relu(self.dense2(X2))

        # Concatenate input streams
        X = torch.cat([X1, X2], 1)

        # V stream
        V = F.relu(self.V_dense(X))
        V = self.V(V)

        # A stream
        A = F.relu(self.A_dense(X))
        A = self.A(A)

        return V, A
    
    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.checkpoint_file)

    


class DRL_algorithm:

    def __init__(self, memory, replay_exp_initial_condition):

        self.training_finished = False
        self.update_network_counter = 1
        
        # Epsilon greedy exploration parameters
        self.epsilon = 1.0
        self.epsilon_final = 0.01          
        self.epsilon_interval = self.epsilon - self.epsilon_final
        self.epsilon_greedy_frames = 750_000

        self.memory = memory
        self.replay_exp_initial_condition = replay_exp_initial_condition        
        
        # Batch size
        self.batch_size = 32
        # Number of action of the agent
        self.action_dim = 5
        
        self.discount = 0.99
        self.mean_loss = 0
        
        self.p_action = 0
        self.same_action_counter = 0

        ####### MODELS        
        # Q network
        self.q_net = duelingDQN(self.action_dim,
                                chkpt_dir='./records/networks',
                                name='duelingDQN')          
        # Q target network
        self.q_target_net = duelingDQN(self.action_dim,
                                       chkpt_dir='./records/networks',
                                       name='target_duelingDQN')              
        # Learning rate decay
        self.q_target_net.load_state_dict(self.q_net.state_dict())

    def policy(self, state, lidar_state):
        if (np.random.uniform(0.0,1.0) < self.epsilon) or (self.memory.experience_ind <= self.replay_exp_initial_condition): 
            # If the random number is less than epsilon
            # Choose a random action
            action = int(np.random.choice(5,1))
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
            # Lidar state
            lidar_states = np.empty((1,4), dtype=int)
            
            lidar_states = np.reshape(lidar_states, (1, len(lidar_state)))            
            lidar_state_tensor = torch.tensor(lidar_states).to(self.q_net.device)    
            # Prediction            
            V, A = self.q_net.forward([img_state_tensor, lidar_state_tensor])                        
            
            A_mean = torch.mean(A)       
            action = torch.argmax((V + (A - A_mean))).item()            
            
            if self.p_action == action:
                self.same_action_counter += 1
            self.p_action = action

            if self.same_action_counter == 30:                
                action = int(np.random.choice(5,1))                                    
            print("Action taken by the network.!!!!!!!!!!!!!!!!!")
            return action
        
    def train(self):
        # Start training condition
        if self.memory.experience_ind <= self.replay_exp_initial_condition:
            self.training_finished = True
            return            
        # Sampling minibatch                
        states_batch, lidar_c_states, rewards_batch, minibatch_actions, next_states_batch, lidar_n_states, dones_batch = \
            self.memory.sample(self.batch_size)
        
        states = torch.tensor(states_batch).to(self.q_net.device)
        lidar_current_states = torch.tensor(lidar_c_states).to(self.q_net.device)
        next_states = torch.tensor(next_states_batch).to(self.q_net.device)
        lidar_next_states = torch.tensor(lidar_n_states).to(self.q_net.device)
        rewards = torch.tensor(rewards_batch).to(self.q_net.device)
        actions = torch.tensor(minibatch_actions).to(self.q_net.device)
        dones = torch.tensor(dones_batch).to(self.q_net.device)
        
        self.q_net.optimizer.zero_grad()

        indices = np.arange(self.batch_size)
        
        V_s, A_s = self.q_net.forward([states, lidar_current_states]) 
        V_s_target, A_s_target = self.q_target_net.forward([next_states, lidar_next_states])
        V_s_eval, A_s_eval = self.q_net.forward([next_states, lidar_next_states])
        
        q_pred = torch.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = torch.add(V_s_target, (A_s_target - A_s_target.mean(dim=1, keepdim=True)))                           
        q_eval = torch.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))   

        max_actions = torch.argmax(q_eval, dim=1)            
        
        q_next[dones] = 0.0
        q_target = rewards + self.discount * q_next[indices, max_actions]

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

        if self.update_network_counter == 10000:
            # Copy weights
            self.q_target_net.load_state_dict(self.q_net.state_dict())
            self.update_network_counter = 1

        self.training_finished = True
        print("Training Finished")

    def save_models(self):
        self.q_net.save_checkpoint()
        self.q_target_net.save_checkpoint()
        print("Models saved")
    
    def load_models(self):        
        self.q_net.load_state_dict(torch.load("model/lunar_lander_model"))          
        self.q_target_net.load_state_dict(torch.load("model/lunar_lander_target_model"))          
        self.q_target_net.load_state_dict(self.q_net.state_dict())  
        print("Models loaded")
