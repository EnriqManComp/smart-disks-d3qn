"""
Author: Enrique Manuel Companioni Valle
Date: 25/04/2021
Description: This script is the main file to run the Deep Reinforcement Learning algorithm to solve the evasor-pursuiter problem.
The algorithm is based on the Deep Q-Learning algorithm. The environment is designed using the Pygame library.
The algorithm is implemented using the TensorFlow library.
The algorithm is trained using the Random Sample Experience Replay method.

The code defines the `Environment` class, which represents the game environment for the Deep Reinforcement Learning algorithm.
It contains methods to initialize the environment, run the game algorithm, and handle the game logic.
The environment consists of a Pygame window, obstacles, players (pursuiter and evasor), and a sensor.
The algorithm uses the Deep Q-Learning algorithm to train the pursuiter to catch the evasor while avoiding obstacles.
The algorithm is trained using the Random Sample Experience Replay method, which stores and replays past experiences to improve learning.
"""
#################################### 
##         Import Libraries       ##
####################################

import pygame
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import gc
import time

from pursuiter import Pursuiter
from evasor import Evasor
from utils import Utils
from obstacles import Obstacles
from random_sample_exp_replay import ReplayBuffer
from DRL_algorithm import DRL_algorithm
from sensors import Sensor

#################################### 
##       Environment Design       ##
####################################

class Environment:
    """
        This class define the pygame functionality to implement the Deep Reinforcement Learning method.
    """
    def __init__(self):        
        ##### PYGAME
        # Initialise pygame
        pygame.init()
        # Set window
        self.screen = pygame.display.set_mode((200, 200))
        # FPS controler
        self.clock = pygame.time.Clock() 

        ##### OBSTACLES
        self.sensor = Sensor()
        self.obstacles = Obstacles(self.screen)

        ##### UTILS
        self.utils = Utils(self.obstacles, self.sensor)

        ##### CONSTANT PARAMETERS
        # Episodes
        self.EPISODES = 10_000
        # Possible Actions
        self.ACTIONS = {
            0: "NO ACTION",            
            1: "UP",
            2: "DOWN",
            3: "LEFT",
            4: "RIGHT",
            5: "DOUBLE-UP",
            6: "DOUBLE-DOWN",
            7: "DOUBLE-LEFT",
            8: "DOUBLE-RIGHT" 
        }
        # Possible rewards
        self.REWARDS = {
            "COLLISION": -100,
            "LIVING PENALTY": -1.0,
            "GOAL": +100,
            "GOAL-COLLISION-EVASOR": -25.0
        }       

        ###### MEMORY
        # Initial number of experience in storage to start the training
        self.replay_exp_initial_condition = 50_000
        self.memory = ReplayBuffer(max_size=1_000_000, pointer=0, screen=self.screen)

        ###### Deep Reinforcement Learning algorithm
        self.drl_algorithm = DRL_algorithm(self.memory, self.replay_exp_initial_condition)

        # Run parameters
        self.current_state = np.empty((200,200,3))
        self.next_state = np.empty((200,200,3))        
        self.done = False

        # Metric parameters        
        self.record_scores = []        

        #### Players
        self.pursuiter = Pursuiter()
        self.evasor = Evasor()

        ## triggers
        self.restart_params_trigger = False
        self.start_new_cycle_trigger = True
        self.save_exp_trigger = False
        self.training_trigger = False


    def run(self):              
        """
            This method run the game algorithm
        """

        ######## TRIGGERS
        
       

        save_net_indicator = 1        
        
        load = False
        if load:                                   
            self.drl_algorithm.epsilon = 0.6473956600076547           
            self.memory.experience_ind = 344000            
            self.memory.storage = 344000
            self.drl_algorithm.load_net(293)
            save_net_indicator = 294
            
        for epis in range(1, self.EPISODES+1):
            ##### Restart the initial parameters in each episode
            self.done = False           
            
            reward = 0
            
            scores = 0.0
            run_time = 1           

            ##### TRIGGERS
            # 1st step
            self.restart_params_trigger = False
            # 2nd step
            self.start_new_cycle_trigger = True
            # 3rd step
            self.save_exp_trigger = False
            # 4th step
            self.training_trigger = False

            # Spawn the agents and draw the environment
            self.screen.fill((138,138,138))
            self.obstacles.render_walls()

            print("STARTING RESPAWN")

            self.evasor.position = []
            self.pursuiter.position = []
            self.sensor.position = []
            x, y = self.utils.random_spawn(evasor_spawn=True)
            self.evasor.position.append(x)            
            self.evasor.position.append(y)
            self.evasor.spawn(self.screen)

            x, y = self.utils.random_spawn(self.evasor.position, self.screen, self.evasor, evasor_spawn=False)
            self.pursuiter.position.append(x)            
            self.pursuiter.position.append(y)
            self.pursuiter.spawn(self.screen)

            self.sensor.position.append(self.pursuiter.position[0])
            self.sensor.position.append(self.pursuiter.position[1])
            
            self.sensor.lidar(self.screen)
            self.screen.fill((138,138,138))
            self.obstacles.render_walls()
            self.evasor.spawn(self.screen)
            self.pursuiter.spawn(self.screen)         
            # Update the screen
            pygame.display.update()             
            
            
                           
            print("######################################################EPISODE: ", epis)
            while (run_time <= 600) and (not self.done):
                # Run the game algorithm
                print("Run time: ", run_time)
                for event in pygame.event.get():
                    # Exit event
                    if event.type == pygame.QUIT:
                        sys.exit()
                    
                ########### Restart parameters
                        
                if self.restart_params_trigger:                    
                    ### Save metrics
                    # Save score
                    scores += reward
                                                               
                    # Check end episode condition
                    if (self.done) or (scores == 1000):
                        break
                    # Increase run time 
                    run_time += 1      
                    # Set the dimensions of the states          
                    self.current_state = np.empty((200,200,3))
                    self.next_state = np.empty((200,200,3))        
                     
                    gc.collect()
                    time.sleep(1)                  
                    # Restart the trigger cycle
                    self.drl_algorithm.training_finished = False
                    self.restart_params_trigger = False                
                    self.start_new_cycle_trigger = True
                
                ########### CAPTURE CURRENT STATE
                    
                if self.start_new_cycle_trigger:
                    for _ in range(4):                
                        # Get current state
                        self.current_state = self.get_capture()                    
                        
                        ###### EXECUTE THE ACTION 
                        # Get lidar observation
                        lidar_current_state = self.utils.lidar_observations(self.pursuiter.position[0], self.pursuiter.position[1], self.evasor)
                        # Select an action
                        action = self.drl_algorithm.policy(self.current_state, lidar_current_state)  
                        action = self.ACTIONS[action]                
                        print(action)               
                        # Execute the action selected
                        self.pursuiter.controls(action= action)                               
                        # Update the position of the lidar rectangles
                        self.sensor.update_position(self.pursuiter.position)

                        ######## DRAW ZONE
                        # Draw the window with the updates
                        # Draw lidar
                        self.sensor.lidar(self.screen)
                        # Draw background
                        self.screen.fill((138,138,138))
                        # Draw obstacles                    
                        self.obstacles.render_walls()
                        # Draw pursuiter
                        self.pursuiter.spawn(self.screen)                         
                        # Draw evasor
                        self.evasor.spawn(self.screen)                                                           
                        
                        ####### END DRAW ZONE
                    
                        ###########  CAPTURE NEXT STATE 
                        self.next_state = self.get_capture()
                        # Get lidar next state observations
                        lidar_next_state = self.utils.lidar_observations(self.pursuiter.position[0], self.pursuiter.position[1], self.evasor)                                
                        
                        ######### CHECK COLLISIONS
                                            
                        pursuiter_collision_condition = self.utils.collision(self.pursuiter.robot, self.evasor.robot, self.pursuiter.position, self.evasor.position)
                        print(pursuiter_collision_condition)
                        if pursuiter_collision_condition == "COLLISION" or pursuiter_collision_condition == "GOAL-COLLISION-EVASOR":
                            self.done = True           
                        
                        ######## GET REWARD
                        reward = self.REWARDS[pursuiter_collision_condition]
                        
                        # Activate and deactivate step triggers
                        self.save_exp_trigger = True                   
                    
                    ########## SAVE EXPERIENCE 
                    
                    if self.save_exp_trigger:      
                        # Add experience in memory                   
                        experience = [self.current_state.copy(), lidar_current_state, action, reward, self.next_state.copy(), lidar_next_state, self.done]
                        self.memory.add(experience)          
                        
                        # Next step (TRAINING NETWORK)                                               
                        self.save_exp_trigger = False                         
                    
                    print("Training step")
                    self.start_new_cycle_trigger = False
                    self.training_trigger = True                                                

                if self.training_trigger:                
                    # Train the model
                    self.drl_algorithm.train()           
                    # Wait until the training has been completed.
                    while not self.drl_algorithm.training_finished:
                        continue
                    self.restart_params_trigger = True
           
                # Update the display
                pygame.display.update()                
            
            print("END RUN TIME")
            # Save scores of the episode
            self.record_scores.append(scores)            
            
            
            # Check if the algorithm is still in the non-training phase.
            if self.memory.storage >= self.drl_algorithm.replay_exp_initial_condition:
                # Save the model if the score of the episode is grater than 2500
                if (scores >= 500):
                    # Save the weights
                    self.drl_algorithm.save_model(save_net_indicator)
                    # Save network records
                    with open("./records/save_network.txt", 'a') as file:
                        file.write("Save: {0}, Episode: {1}/{2}, Score: {3}, Epsilon: {4}, Play_time: {5}, Spawn distance: {6}\n".format(save_net_indicator, epis, self.EPISODES, scores, self.drl_algorithm.epsilon, run_time, self.utils.spawn_eucl_dist))
                    save_net_indicator += 1
            # Save records and model each 100 episodes.
            if epis % 100 == 0:        
                # Save scores and losses
                with open("./records/save_records.txt", 'a') as file:
                    file.write("Scores: {0}\n".format(self.record_scores))
                
                # Save memory data
                data_ptr = pd.Series(self.memory.experience_ind)
                data_ptr.to_csv("./records/data_ptr.csv", index=False, header=None, mode='w')

                
                # Restart the record scores and losses
                self.record_scores = []               
            if epis % 500 == 0:
                # Save model record
                with open("./records/save_network.txt", 'a') as file:
                    file.write("Save: {0}, Episode: {1}/{2}, Score: {3}, Epsilon: {4}, Play_time: {5}, Spawn distance: {6}\n".format(save_net_indicator, epis, self.EPISODES, scores, self.drl_algorithm.epsilon, run_time, self.utils.spawn_eucl_dist))
                # Save the weights of the model
                self.drl_algorithm.save_model(save_net_indicator)
                # Increase the save model counter
                save_net_indicator += 1

            gc.collect()
            time.sleep(1)
            print("END EPISODE")

        pygame.quit()

    def get_capture(self):
        """
            Capture the current state image.
            Return:
                captured (NumPy array): Return a NumPy array image with dimensions (405, 410, 3).

        """
        # Create the Surface        
        capture = pygame.Surface((self.screen.get_width(), self.screen.get_height()))
        # Blit the screen on the Surface
        capture.blit(self.screen, (0, 0))
        # Convert from Surface to Array
               
        # Return the capture
        return pygame.surfarray.array3d(capture)    


# Run the algorithm
# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')
train = True
gc.enable()
gc.collect()
if train:
    Environment().run()

