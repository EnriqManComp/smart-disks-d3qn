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
    

    


    




def duelingDQN(input_dim, distance_sensor, fc1_units, fc2_units, action_dim, lr):
    X_input = tf.keras.layers.Input(input_dim)
    X2_input = tf.keras.layers.Input(distance_sensor)
    X = X_input
    X2 = X2_input

    X = tf.keras.layers.Conv2D(filters=64,kernel_size=(8,8),strides=4)(X)
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.Conv2D(filters=32,kernel_size=(4,4),strides=2)(X) 
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=1)(X)
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.Flatten()(X)

    X2 = tf.keras.layers.Dense(units=256)(X2)    

    X_X2 = tf.keras.layers.Concatenate(axis=1)([X, X2]) 

    V = tf.keras.layers.Dense(units=fc1_units)(X_X2)
    V = tf.keras.layers.ReLU()(V)
    V = tf.keras.layers.Dense(1, activation=None)(V)
    V = tf.keras.layers.Lambda(lambda s: backend.expand_dims(s[:, 0], -1), output_shape=(action_dim,))(V)
    
    A = tf.keras.layers.Dense(units=fc2_units)(X_X2)
    A = tf.keras.layers.ReLU()(A)
    A = tf.keras.layers.Dense(action_dim, activation=None)(A)
    A = tf.keras.layers.Lambda(lambda a: a[:, :] - backend.mean(a[:, :], keepdims=True), output_shape=(action_dim,))(A)
    

    X_X2 = tf.keras.layers.Add()([V, A])
    
    model = tf.keras.Model(inputs = [X_input, X2_input], outputs = X_X2)    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse') 
    model.summary()   
    
    return model

class DRL_algorithm:

    def __init__(self, memory, replay_exp_initial_condition):

        self.training_finished = False
        self.update_network_counter = 1
        
        # Epsilon greedy exploration parameters
        self.epsilon = 1.0
        self.epsilon_final = 0.1          

        self.epsilon_interval = 1.0 - 0.1
        self.epsilon_greedy_frames = 300_000.0



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
        self.mse_loss = 0

        self.action = 0
        self.same_action_counter = 0

        ####### MODELS
        self.loss_function = tf.keras.losses.Huber()
        #self.loss_function = nn.HuberLoss()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        # Q network
        #self.q_net = duelingDQN(input_dim, 4, fc1_units, fc2_units, self.action_dim, lr)
        self.q_net = dqn_network(4, self.action_dim)        
        self.q_net.compile(optimizer=self.optimizer, loss=self.loss_function)
        #visualkeras.layered_view(self.q_net, legend=True, to_file='model.png') # write to disk   
        # Q target network
        self.q_target_net = copy.deepcopy(self.q_net)
        # Copy weights
        self.q_target_net.set_weights(self.q_net.get_weights())        
    
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
            actions = self.q_net.predict([img_state, lidar_states])            
            '''
            # Epsilon greedy exploration in the network output
            if np.random.uniform(0.0,1.0) < self.epsilon_action:
                # Select the second max Q value                
                # Sorting indices                
                sorted_indices = tf.argsort(actions, direction='DESCENDING')                
                # Get the second max Q value index = action                
                action = sorted_indices[0, 1].numpy()                
                # Counter of repeated action in the previous step
                if self.action == action:
                    self.same_action_counter += 1
                self.action = action               
            else:                 
                # Selecting the action with max Q value
                action = tf.math.argmax(actions, axis=1).numpy()[0]
                # Counter of repeated action in the previous step
                if self.action == action:
                    self.same_action_counter += 1
                self.action = action
            '''
            
            action = tf.math.argmax(actions, axis=1).numpy()[0]
                # Counter of repeated action in the previous step
            if self.action == action:
                self.same_action_counter += 1
            self.action = action

            if self.same_action_counter == 30:                
                action = int(np.random.choice(4,1))                                    

            # Update epsilon
            #self.epsilon_action = self.epsilon_action - self.epsilon_action_decay if self.epsilon_action > self.epsilon_action_final else self.epsilon_action_final
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
        
        #q_preds = self.q_net([states_batch, lidar_c_states])         
        # Select the max Qs of the samples
        #q_preds = tf.math.reduce_max(q_preds, axis=1).numpy()       
        q_preds = self.q_net([states_batch, lidar_c_states])
        # Select the actions in each next states of the samples
        q_eval = self.q_net([next_states_batch, lidar_n_states])
        next_max_actions = np.argmax(q_eval, axis=1)
        # Predictions with Q target network
        q_next_preds = self.q_target_net([next_states_batch, lidar_n_states])        
        
        # Create variables for the next steps
        q_target = np.copy(q_preds)               
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
        '''
        masks = tf.one_hot(minibatch_actions, self.action_dim)                
        

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values                        
            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_preds, masks), axis=1)                      
            
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss_function(q_target, q_action)
        
        # Backpropagation
        grads = tape.gradient(loss, self.q_net.trainable_variables)
        
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))

        print("HEREEEEEEEEEEEEE")

        '''
        #metrics = self.q_net.train_on_batch(x= [states_batch, lidar_c_states], y= q_target, sample_weight=minibatch_weights, return_dict=True)
        metrics = self.q_net.train_on_batch(x= [states_batch, lidar_c_states], y= q_target, return_dict=True)
        # Update priorities
        #self.memory.update(minibatch_tree_idx, errors)        
        # Save loss metric
        
        
        self.mse_loss = np.mean(metrics['loss'])
        

        
        # Update epsilon
         # Decay probability of taking random action
        self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
        self.epsilon = max(self.epsilon, self.epsilon_final)
        #self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_final else self.epsilon_final
        # Polyak's Average
        #self.soft_update()
        self.update_network_counter += 1

        if self.update_network_counter == 8000:
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
    