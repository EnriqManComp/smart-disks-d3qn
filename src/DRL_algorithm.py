import numpy as np
import tensorflow as tf
from keras import backend
import copy
from PIL import Image
#import visualkeras




######## NETWORK ARCHITECTURE



def dqn_network(distance_sensor, action_dim):
    X_input = tf.keras.layers.Input(shape=(405,410,1,))
    X2_input = tf.keras.layers.Input(shape=(4,))

    X = tf.keras.layers.Conv2D(filters=32,kernel_size=(8,8),strides=4)(X_input)
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=2)(X) 
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=1)(X)
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.Flatten()(X)

    #X2 = tf.keras.layers.BatchNormalization(X2_input)
    X2 = tf.keras.layers.Dense(units=256)(X2_input)    
    X2 = tf.keras.layers.ReLU()(X2)
    X_X2 = tf.keras.layers.Concatenate(axis=1)([X, X2])

    OUTPUT = tf.keras.layers.Dense(units=512)(X_X2)
    OUTPUT = tf.keras.layers.ReLU()(OUTPUT)
    OUTPUT = tf.keras.layers.Dense(units=action_dim, activation='linear')(OUTPUT)

    return tf.keras.Model(inputs = [X_input, X2_input], outputs = OUTPUT)    
    

class duelingDQN(tf.keras.Model):
    def __init__(self,action_dim):
        super(duelingDQN, self).__init__()
        # 1st Input stream        
        self.conv1 = tf.keras.layers.Conv2D(filters=32,kernel_size=(8,8),strides=4, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=2, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=1, activation='relu')
        self.flatten = tf.keras.layers.Flatten()

        # 2nd Input stream
        self.dense1 = tf.keras.layers.Dense(units=256, activation='relu')

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
        X2 = self.dense1(state[1])

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
        X2 = self.dense1(state[1])

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

        self.epsilon_interval = 1.0 - self.epsilon_final
        self.epsilon_greedy_frames = 1_000_000.0



        self.epsilon_action = 1.0
        self.epsilon_action_final = 0.05
        self.epsilon_action_decay = 0.0005  

        self.memory = memory
        self.replay_exp_initial_condition = replay_exp_initial_condition
        ####### TESTING PARAMETERS
        # Polyak's parameter
        self.tau = 0.001
        # Learning rate
        #lr = 0.00025 
        # Number of V Stream Units or Neuron
        #fc1_units = 256
        # Number of A Stream Units or Neuron
        #fc2_units = 256
        # Batch size
        self.batch_size = 32
        # Number of action of the agent
        self.action_dim = 5
        # Dimension of the captured states
        # Width, height, dimension        
        #input_dim = [405,410,1]

        self.discount = 0.99
        self.mean_loss = 0

        self.action = 0
        self.same_action_counter = 0

        ####### MODELS
        self.loss_function = tf.keras.losses.Huber()
        #self.loss_function = nn.HuberLoss()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00015)
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
        self.lr_decay = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.9,
            patience=2,           
            min_lr=0.000001,            
            )
        self.episodes = 1

    
    def policy(self, state, lidar_state):
        if (np.random.uniform(0.0,1.0) < self.epsilon) or (self.memory.experience_ind <= self.replay_exp_initial_condition): 
            # If the random number is less than epsilon
            # Choose a random action
            action = int(np.random.choice(4,1))
            # Counter of repeated action in the previous step
            if self.action == action:
                self.same_action_counter += 1
            self.action = action
                                            
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
            if self.action == action:
                self.same_action_counter += 1
            self.action = action

            if self.same_action_counter == 30:                
                action = int(np.random.choice(4,1))                                    

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
                
        #minibatch_actions_tensor = tf.convert_to_tensor(minibatch_actions, dtype=tf.int32) 
        #print(minibatch_actions_tensor)
        #q_preds = self.q_net([states_batch, lidar_c_states])         
        # Select the max Qs of the samples
        #q_preds = tf.math.reduce_max(q_preds, axis=1).numpy()       
        q_preds = self.q_net([states_batch, lidar_c_states])        
         
        #print(type(minibatch_actions))
        #print(minibatch_actions)
        #print(q_preds)
        #print(tf.Tensor(minibatch_actions))
        #q_preds = q_preds[indices, tf.Tensor(minibatch_actions)]
        #print(q_preds)
        # Select the actions in each next states of the samples
        q_eval = self.q_net([next_states_batch, lidar_n_states])
        next_max_actions = np.argmax(q_eval, axis=1)
        # Predictions with Q target network
        q_next_preds = self.q_target_net([next_states_batch, lidar_n_states])        
        
        # Create variables for the next steps
        q_target = q_preds.numpy()             
        #errors = np.empty_like(next_actions)              
        
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
            #errors[idx] = np.abs(q_preds[idx] - q_target[idx])
        
        print("Training... ")        
        # Training network
         # Create a mask so we only calculate loss on the updated Q-values        
        
        #metrics = self.q_net.train_on_batch(x= [states_batch, lidar_c_states], y= q_target, sample_weight=minibatch_weights, return_dict=True)
        metrics = self.q_net.train_on_batch(x= [states_batch, lidar_c_states], y= q_target, return_dict=True)
        # Update priorities
        #self.memory.update(minibatch_tree_idx, errors)        
        # Save loss metric
        #self.lr_decay.on_epoch_end(self.episodes)
        #config = self.q_net.optimizer.get_config()
        #print(config['learning_rate'])
        
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
        self.q_net.save(root_path+'/{0}'.format(f))   
        

    def load_net(self):
        model_path = 'C:/Carpeta personal/school/Thesis/code/records/model_used'        
        self.q_net = tf.keras.models.load_model(model_path)       
        # Copy weights
        self.q_target_net.set_weights(self.q_net.get_weights())
        print("Network loaded!!!!!!!!!")
    