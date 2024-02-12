"""
Developed by: Enrique Manuel Companioni Valle
"""
#################################### 
##         Import Libraries       ##
####################################
import pygame
import sys
import pandas as pd
import numpy as np

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
        self.screen = pygame.display.set_mode((405, 410))
        # FPS controler
        self.clock = pygame.time.Clock() 

        ##### OBSTACLES
        self.sensor = Sensor()
        self.obstacles = Obstacles(self.screen)

        ##### UTILS
        self.utils = Utils(self.obstacles, self.sensor)

        ##### CONSTANT PARAMETERS
        # Possible Actions
        self.ACTIONS = {
            0: "NO ACTION",            
            1: "UP",
            2: "DOWN",
            3: "LEFT",
            4: "RIGHT" 
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
        self.replay_exp_initial_condition = 33
        self.memory = ReplayBuffer(max_size=1_000_000, pointer=0, screen=self.screen)

        ###### Deep Reinforcement Learning algorithm
        self.drl_algorithm = DRL_algorithm(self.memory, self.replay_exp_initial_condition)

    def run(self):              
        """
            This method run the game algorithm
        """

        ######## ALGORITHM PARAMETERS

        done = False
        counter = 0
        current_state = np.empty((405,410,3))
        next_state = np.empty((405,410,3))
        reward = 0
        losses = []
        scores = 0.0
        run_time = 1
        train_counter = 1

        ######## TRIGGERS
        
        # 1st step
        restart_params_trigger = False
        # 2nd step
        start_new_cycle_trigger = True
        # 3rd step
        current_state_trigger = False
        # 4th step
        action_delay = False
        # 5th step
        next_state_trigger = False
        # 6th step
        collision_trigger = False
        # 7th step
        save_exp_trigger = False
        # 8th step
        training_trigger = False
        
        ##### PLAYERS

        # Pursuiter
        pursuiter = Pursuiter()
        # Evasor
        evasor = Evasor()              

        EPISODES = 10000
        save_net_indicator = 1
        record_scores = []
        record_losses = []

        #self.memory.experience_ind = 15362
        #self.memory.tree.data_pointer = 15362
        #self.memory.load()
        #self.memory.experience_ind = 159265
        load = False
        if load:
            self.drl_algorithm.load_net()
            self.drl_algorithm.epsilon = 1.0
            #self.drl_algorithm.epsilon_action = 0.05
            self.memory.experience_ind = 159265
            #self.memory.tree.data_pointer = 50001
            #self.memory.load()
            save_net_indicator = 139
            #self.memory.PER_b = 0.1

        for epis in range(1, EPISODES+1):
            ##### Restart the initial parameters in each episode
            done = False
            counter = 0
            current_state = np.empty((405,410,3))
            next_state = np.empty((405,410,3))
            reward = 0
            losses = []
            scores = 0.0
            run_time = 1
            train_counter = 1

            ##### TRIGGERS
            # 1st step
            restart_params_trigger = False
            # 2nd step
            start_new_cycle_trigger = True
            # 3rd step
            current_state_trigger = False
            # 4th step
            action_delay = False
            # 5th step
            next_state_trigger = False
            # 6th step
            collision_trigger = False
            # 7th step
            save_exp_trigger = False
            # 8th step
            training_trigger = False

            # Spawn the agents and draw the environment
            self.screen.fill((138,138,138))
            self.obstacles.render_walls()

            print("STARTING RESPAWN")

            evasor.position = []
            pursuiter.position = []
            self.sensor.position = []
            x, y = self.utils.random_spawn(evasor_spawn=True)
            evasor.position.append(x)            
            evasor.position.append(y)
            evasor.spawn(self.screen)

            x, y = self.utils.random_spawn(evasor.position, self.screen, evasor, evasor_spawn=False)
            pursuiter.position.append(x)            
            pursuiter.position.append(y)
            pursuiter.spawn(self.screen)

            self.sensor.position.append(pursuiter.position[0])
            self.sensor.position.append(pursuiter.position[1])
            
            self.sensor.lidar(self.screen)
            self.screen.fill((138,138,138))
            self.obstacles.render_walls()
            evasor.spawn(self.screen)
            pursuiter.spawn(self.screen)         
            # Update the screen
            pygame.display.update()            

            print("EPISODE: ", epis)
            while (run_time <= 100) and (not done):
                # Run the game algorithm
                print("Run time: ", run_time)
                for event in pygame.event.get():
                    # Exit event
                    if event.type == pygame.QUIT:
                        sys.exit()
                    
                ########### Restart parameters
                        
                if restart_params_trigger:                    
                    ### Save metrics
                    # Save score
                    scores += reward
                    # Save losses in each training steps
                    if self.memory.experience_ind > self.replay_exp_initial_condition:                      
                        losses.append(self.drl_algorithm.mean_loss)                        
                    # Check end episode condition
                    if done:
                        break
                    # Increase run time 
                    run_time += 1      
                    # Set the dimensions of the states          
                    current_state = np.empty((405,410,3))
                    next_state = np.empty((405,410,3))
                    # Restart the loss of the DRL algorithm                 
                    self.drl_algorithm.mean_loss = []                    
                    # Restart the trigger cycle
                    self.drl_algorithm.training_finished = False
                    restart_params_trigger = False                
                    start_new_cycle_trigger = True
                
                ########### CAPTURE CURRENT STATE
                    
                if start_new_cycle_trigger:                
                    # Get current state
                    current_state = self.get_capture()
                
                    ###### EXECUTE THE ACTION 
                    # Get lidar observation
                    lidar_current_state = self.utils.lidar_observations(pursuiter.position[0], pursuiter.position[1], evasor)
                    # Select an action
                    action = self.drl_algorithm.policy(current_state, lidar_current_state)  
                    action = self.ACTIONS[action]                
                    print(action)               
                    # Execute the action selected
                    pursuiter.controls(action= action)                               
                    # Update the position of the lidar rectangles
                    self.sensor.update_position(pursuiter.position)

                    ######## DRAW ZONE
                    # Draw the window with the updates
                    # Draw lidar
                    self.sensor.lidar(self.screen)
                    # Draw background
                    self.screen.fill((138,138,138))
                    # Draw obstacles                    
                    self.obstacles.render_walls()
                    # Draw pursuiter
                    pursuiter.spawn(self.screen)                         
                    # Draw evasor
                    evasor.spawn(self.screen)                                                           
                    
                    ####### END DRAW ZONE
                
                    ###########  CAPTURE NEXT STATE 
                    next_state = self.get_capture()
                    # Get lidar next state observations
                    lidar_next_state = self.utils.lidar_observations(pursuiter.position[0], pursuiter.position[1], evasor)                                
                    
                    ######### CHECK COLLISIONS
                                        
                    pursuiter_collision_condition = self.utils.collision(pursuiter.robot, evasor.robot, pursuiter.position, evasor.position)
                    print(pursuiter_collision_condition)
                    if pursuiter_collision_condition == "COLLISION" or pursuiter_collision_condition == "GOAL-COLLISION-EVASOR":
                        done = True           
                    
                    ######## GET REWARD
                    reward = self.REWARDS[pursuiter_collision_condition]
                    
                    # Activate and deactivate step triggers
                    save_exp_trigger = True
                    collision_trigger = False           
                
                ########## SAVE EXPERIENCE 
                
                if save_exp_trigger:      
                    # Add experience in memory                   
                    experience = [current_state, lidar_current_state, action, reward, next_state, lidar_next_state, done]
                    self.memory.add(experience)          
                    
                    # Next step (TRAINING NETWORK)                                               
                    save_exp_trigger = False 
                    start_new_cycle_trigger = False
                    training_trigger = True                    

                if self.drl_algorithm.training_finished:                                 
                    # Wait until the training has been completed.
                    restart_params_trigger = True            

                if training_trigger:                
                    # Train the model
                    self.drl_algorithm.train()           
                    training_trigger = False                             
                
                # Update the display
                pygame.display.update()
                self.clock.tick(60)
            
            print("END RUN TIME")
            # Save scores of the episode
            record_scores.append(scores)
            # Fill with 0.0 the empty losses
            if len(losses) == 0:
                record_losses.append(0.0)
            else:
                # Save the mean of the episode losses                                  
                record_losses.append(np.mean(losses))
            # Restart the losses variable
            losses = []
            # Check if the algorithm is still in the non-training phase.
            if self.memory.storage >= self.drl_algorithm.replay_exp_initial_condition:
                # Save the model if the score of the episode is grater than 2500
                if (scores >= 2500):
                    # Save the weights
                    self.drl_algorithm.save_results('./records/networks', save_net_indicator)
                    # Save network records
                    with open("./records/save_network.txt", 'a') as file:
                        file.write("Save: {0}, Episode: {1}/{2}, Score: {3}, Epsilon: {4}, Epsilon_action: {5}, Play_time: {6}, Spawn distance: {7}\n".format(save_net_indicator, epis, EPISODES, scores, self.drl_algorithm.epsilon, self.drl_algorithm.epsilon_action, run_time, self.utils.spawn_eucl_dist))
                    save_net_indicator += 1
            # Save records and model each 100 episodes.
            if epis % 100 == 0:        
                # Save scores and losses
                with open("./records/save_records.txt", 'a') as file:
                    file.write("Scores: {0}, Losses: {1}\n".format(record_scores, record_losses))
                # Save model record
                with open("./records/save_network.txt", 'a') as file:
                    file.write("Save: {0}, Episode: {1}/{2}, Score: {3}, Epsilon: {4}, Epsilon_action: {5}, Play_time: {6}, Spawn distance: {7}\n".format(save_net_indicator, epis, EPISODES, scores, self.drl_algorithm.epsilon, self.drl_algorithm.epsilon_action, run_time, self.utils.spawn_eucl_dist))
                # Save memory data
                data_ptr = pd.Series(self.memory.experience_ind)
                data_ptr.to_csv("./records/data_ptr.csv", index=False, header=None, mode='w')

                # Save the weights of the model
                self.drl_algorithm.save_results('./records/networks', save_net_indicator)
                # Increase the save model counter
                save_net_indicator += 1
                # Restart the record scores and losses
                record_scores = []
                record_losses = []
                        
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
        captured = pygame.surfarray.array3d(capture)
        # Return the capture
        return captured         
            
    
# Run the algorithm
Environment().run()
