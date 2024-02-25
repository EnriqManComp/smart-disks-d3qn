import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
#import visualkeras




######## NETWORK ARCHITECTURE

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
    
    

class DRL_algorithm:

    def __init__(self, memory, replay_exp_initial_condition):

        self.training_finished = False
        self.update_network_counter = 1
        
        # Epsilon greedy exploration parameters
        self.epsilon = 1.0
        self.epsilon_final = 0.01          

        self.epsilon_interval = self.epsilon - self.epsilon_final
        self.epsilon_greedy_frames = 500_000



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
        self.loss_function = tf.keras.losses.Huber()
        #self.loss_function = nn.HuberLoss()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        # Q network
        self.q_net = duelingDQN(self.action_dim)
        #self.q_net = dqn_network(4, self.action_dim)        
        self.q_net.compile(optimizer=self.optimizer, loss=self.loss_function)
        #visualkeras.layered_view(self.q_net, legend=True, to_file='model.png') # write to disk   
        # Q target network
        self.q_target_net = duelingDQN(self.action_dim)
        self.q_target_net.compile(optimizer=self.optimizer, loss=self.loss_function)
        # Copy weights
        self.q_target_net.set_weights(self.q_net.get_weights())        
        # Learning rate decay
        
        self.episodes = 1

    
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
            #state = np.empty((state_list[0].shape[0], state_list[0].shape[1]))
            #for i, image in enumerate(state_list):   
            img_state = Image.fromarray(state).convert('L') 
            img_state = img_state.rotate(-90)   
            img_state = img_state.transpose(Image.FLIP_LEFT_RIGHT)          
                #image_temp = np.array(image_temp) / 255.0
            img_state = np.array(img_state) / 255.0            

            # Expand dimensions 
            img_state = np.expand_dims(img_state, axis = 0)
            # Lidar state
            lidar_states = np.empty((1,4), dtype=int)
            
            lidar_states = np.reshape(lidar_states, (1, len(lidar_state)))            
                
            # Prediction
            A = self.q_net.advantage([img_state, lidar_states])                        
            
            action = tf.math.argmax(A, axis=1).numpy()[0]
                # Counter of repeated action in the previous step
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
        #minibatch_weights, minibatch_tree_idx, states_batch, lidar_c_states, rewards_batch, minibatch_actions, next_states_batch, lidar_n_states, dones_batch = self.memory.sample(self.batch_size)
        states_batch, lidar_c_states, rewards_batch, minibatch_actions, next_states_batch, lidar_n_states, dones_batch = \
            self.memory.sample(self.batch_size)
        # Prediction with Q network     
                
           
        q_preds = self.q_net([states_batch, lidar_c_states])        
         
        
        # Select the actions in each next states of the samples
        q_eval = self.q_net([next_states_batch, lidar_n_states])
        next_max_actions = np.argmax(q_eval, axis=1)
        # Predictions with Q target network
        q_next_preds = self.q_target_net([next_states_batch, lidar_n_states])        
        
        # Create variables for the next steps
        q_target = q_preds.numpy()             
                    
        
        # Double DQN algorithm
        for idx in range(dones_batch.shape[0]):
            q_target_value = rewards_batch[idx]                     
            if not dones_batch[idx]:                        
                # If not done episode -> evaluate actions predicted
                q_action_evaluated = q_next_preds[idx][next_max_actions[idx]]
                # Bellman equation variation for Double DQN algorithm                                                
                q_target_value += self.discount*q_action_evaluated               
            # Final Q from Bellman Equation
            q_target[idx][minibatch_actions[idx]] = q_target_value
            # Compute errors between q_preds and 
            
        
        print("Training... ")        
        # Training network
         # Create a mask so we only calculate loss on the updated Q-values        
        
        #metrics = self.q_net.train_on_batch(x= [states_batch, lidar_c_states], y= q_target, sample_weight=minibatch_weights, return_dict=True)
        metrics = self.q_net.train_on_batch(x= [states_batch, lidar_c_states], y= q_target, return_dict=True)
        # Update priorities
        
        
        self.mean_loss = np.mean(metrics['loss'])
        

        
        # Update epsilon
         # Decay probability of taking random action
        self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
        self.epsilon = max(self.epsilon, self.epsilon_final)
        #self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_final else self.epsilon_final
        # Polyak's Average
        #self.soft_update()
        self.update_network_counter += 1
        self.episodes += 1

        if self.update_network_counter == 200:
            # Copy weights
            self.q_target_net.set_weights(self.q_net.get_weights())
            self.update_network_counter = 1

        self.training_finished = True
        print("Training Finished")



    def soft_update(self):
        # Update the target model slowly
        new_weights = [self.tau * current + (1 - self.tau) * target for current, target in zip(self.q_net.get_weights(), self.q_target_net.get_weights())]
        self.q_target_net.set_weights(new_weights)

    def save_results(self, root_path, f):
        self.q_net.save_weights(root_path+'/{0}.h5'.format(f))   
        

    def load_net(self, f):
        model_path = './records/networks/'           
        # One call
        batch = np.random.choice(33, 32, replace=False)
        
        minibatch_lidar_c_state = np.empty((32, 4))
        minibatch_current_state = np.empty((32, 200, 200))

        for i, data_index in enumerate(batch):
            ### Get data
            # Getting current state            
            minibatch_current_state[i, :, :] = np.array(Image.open(f"./dataset/{data_index}/current_state/c_s.png")) / 255.0                                
            lidar_current_state = np.array(pd.read_csv(f"./dataset/{data_index}/current_state/lidar.csv", header=0))
            minibatch_lidar_c_state[i, :] = lidar_current_state
            # Getting next state          
            
        ### Activate layers inputs
        q_preds = self.q_net([minibatch_current_state, minibatch_lidar_c_state])
        q_next_preds = self.q_target_net([minibatch_current_state, minibatch_lidar_c_state])        
        
        self.q_net.load_weights(model_path+f'{f}.h5')        
        # Copy weights
        self.q_target_net.set_weights(self.q_net.get_weights())
        print("Network loaded!!!!!!!!!")      
        return