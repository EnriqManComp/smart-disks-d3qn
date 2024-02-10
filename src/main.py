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


class Enviroment:

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

        ####### TRIGGERS

        # 1st step
        self.restart_params_trigger = False
        # 2nd step
        self.start_new_cycle_trigger = True
        # 3rd step
        self.current_state_trigger = False
        # 4th step
        self.action_delay = False
        # 5th step
        self.next_state_trigger = False
        # 6th step
        self.collision_trigger = False
        # 7th step
        self.save_exp_trigger = False
        # 8th step
        self.training_trigger = False

        ###### METRIC PARAMETERS

        self.losses = []
        self.scores = 0.0
        self.run_time = 1

        ###### MEMORY

        self.replay_exp_initial_condition = 50000
        self.memory = ReplayBuffer(max_size=1_000_000, pointer=0, screen=self.screen)
        self.drl_algorithm = DRL_algorithm(self.memory, self.replay_exp_initial_condition)

        ###### ALGORITHM PARAMETER
        self.counter = 0
        self.current_state = np.empty((405,410,3))
        self.next_state = np.empty((405,410,3))

    def run(self):       
        

        ######## ALGORITHM PARAMETERS

        done = False
        self.counter = 0
        self.current_state = []
        self.next_state = []
        reward = 0
        self.losses = []
        self.scores = 0.0
        self.run_time = 1
        train_counter = 1

        
        # 1st step
        self.restart_params_trigger = False
        # 2nd step
        self.start_new_cycle_trigger = True
        # 3rd step
        self.current_state_trigger = False
        # 4th step
        self.action_delay = False
        # 5th step
        self.next_state_trigger = False
        # 6th step
        self.collision_trigger = False
        # 7th step
        self.save_exp_trigger = False
        # 8th step
        self.training_trigger = False
        
        # Time controller
                
        
        ##### PLAYERS

        # Pursuiter
        self.pursuiter = Pursuiter()
        # Evasor
        self.evasor = Evasor()              

        EPISODES = 10000
        save_net_indicator = 1
        record_scores = []
        record_losses = []

        #self.memory.experience_ind = 15362
        #self.memory.tree.data_pointer = 15362
        #self.memory.load()
        self.memory.experience_ind = 159265
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

            done = False
            self.counter = 0
            self.current_state = np.empty((405,410,3))
            self.next_state = np.empty((405,410,3))
            reward = 0
            self.losses = []
            self.scores = 0.0
            self.run_time = 1
            train_counter = 1


            
            # 1st step
            self.restart_params_trigger = False
            # 2nd step
            self.start_new_cycle_trigger = True
            # 3rd step
            self.current_state_trigger = False
            # 4th step
            self.action_delay = False
            # 5th step
            self.next_state_trigger = False
            # 6th step
            self.collision_trigger = False
            # 7th step
            self.save_exp_trigger = False
            # 8th step
            self.training_trigger = False


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
            
            pygame.display.update()

            #self.memory.update_sampling()

            print("EPISODE: ", epis)
            while (self.run_time <= 100) and (not done):
                print("Run time: ", self.run_time)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    

                ########### Restart parameters
                        
                if self.restart_params_trigger:
                    
                    # Save metrics
                    self.scores += reward
                    
                    if self.memory.experience_ind > self.replay_exp_initial_condition:                      
                        if train_counter == 4:
                            self.losses.append(self.drl_algorithm.mse_loss)
                            train_counter = 0                   
                    
                    if done:
                        break

                    self.run_time += 1                
                    self.current_state = np.empty((405,410,3))
                    self.next_state = np.empty((405,410,3))                
                    self.drl_algorithm.mse_loss = []
                    train_counter += 1

                    self.drl_algorithm.training_finished = False
                    self.restart_params_trigger = False                
                    self.start_new_cycle_trigger = True
                
                ########### Capture current state
                    
                #self.counter += 1                        
                
                if self.start_new_cycle_trigger:                
                    self.current_state = self.get_capture()
                
                ########## EXECUTE THE ACTION 
                    
                    lidar_current_state = self.utils.lidar_observations(self.pursuiter.position[0], self.pursuiter.position[1], self.evasor)
                    
                    action = self.drl_algorithm.policy(self.current_state, lidar_current_state)  
                    action = self.ACTIONS[action]                
                    print(action)               
                    self.pursuiter.controls(action= action)                               
                
                ######## DRAW ZONE
                    self.sensor.lidar(self.screen)
                    # Draw background
                    self.screen.fill((138,138,138))
                    # Draw obstacles
                    
                    self.obstacles.render_walls()
                    # Draw pursuiter
                    self.pursuiter.spawn(self.screen)                         
                    self.evasor.spawn(self.screen)
                    self.sensor.update_position(self.pursuiter.position)                                        
                    
                    ####### END DRAW ZONE
                
                    ########### Capture Next state and check collision
                    self.next_state = self.get_capture()
                    
                    lidar_next_state = self.utils.lidar_observations(self.pursuiter.position[0], self.pursuiter.position[1], self.evasor)                                
                    
                    ######### Check collisions and REWARD
                                        
                    pursuiter_collision_condition = self.utils.collision(self.pursuiter.robot, self.evasor.robot, self.pursuiter.position, self.evasor.position)
                    print(pursuiter_collision_condition)
                    if pursuiter_collision_condition == "COLLISION" or pursuiter_collision_condition == "GOAL-COLLISION-EVASOR":
                        done = True           
                    
                    reward = self.REWARDS[pursuiter_collision_condition]
                    
                    
                    self.save_exp_trigger = True
                    self.collision_trigger = False           
                
                ########## SAVE EXPERIENCE IN MEMORY                
                
                if self.save_exp_trigger:                                
                    
                    experience = [self.current_state, lidar_current_state, action, reward, self.next_state, lidar_next_state, done]
                    self.memory.add(experience)          
                    
                    # Next step (TRAINING NETWORK)                                               
                    self.save_exp_trigger = False 
                    self.start_new_cycle_trigger = False
                    if (train_counter == 4):
                        self.training_trigger = True
                    else:
                        self.restart_params_trigger = True            


                if self.drl_algorithm.training_finished:                                 
                    self.restart_params_trigger = True            

                if self.training_trigger:                
                    self.drl_algorithm.train()           
                    self.training_trigger = False                             
                
                
                pygame.display.update()
                self.clock.tick(60)
            
            # Save network conditions
            print("END RUN TIME")
            record_scores.append(self.scores)
            
            if len(self.losses) == 0:
                record_losses.append(0.0)
            else:                                   
                record_losses.append(np.mean(self.losses))
            
            self.losses = []
            if self.memory.storage >= self.drl_algorithm.replay_exp_initial_condition:
                if (self.scores >= 2500):
                    # Save the weights
                    self.drl_algorithm.save_results('./records/networks', save_net_indicator)
                    # Save network records
                    with open("./records/save_network.txt", 'a') as file:
                        file.write("Save: {0}, Episode: {1}/{2}, Score: {3}, Epsilon: {4}, Epsilon_action: {5}, Play_time: {6}, Spawn distance: {7}\n".format(save_net_indicator, epis, EPISODES, self.scores, self.drl_algorithm.epsilon, self.drl_algorithm.epsilon_action, self.run_time, self.utils.spawn_eucl_dist))
                    save_net_indicator += 1
            # Save record frequency 
            if epis % 100 == 0:        
                # Save metric records
                with open("./records/save_records.txt", 'a') as file:
                    file.write("Scores: {0}, Losses: {1}\n".format(record_scores, record_losses))
                
                with open("./records/save_network.txt", 'a') as file:
                    file.write("Save: {0}, Episode: {1}/{2}, Score: {3}, Epsilon: {4}, Epsilon_action: {5}, Play_time: {6}, Spawn distance: {7}\n".format(save_net_indicator, epis, EPISODES, self.scores, self.drl_algorithm.epsilon, self.drl_algorithm.epsilon_action, self.run_time, self.utils.spawn_eucl_dist))

                data_ptr = pd.Series(self.memory.experience_ind)
                data_ptr.to_csv("./records/data_ptr.csv", index=False, header=None, mode='w')

                # Save the weights
                self.drl_algorithm.save_results('./records/networks', save_net_indicator)

                save_net_indicator += 1
                record_scores = []
                record_losses = []
                        
            print("END EPISODE")

        pygame.quit()

    def get_capture(self):
        # Create the Surface        
        capture = pygame.Surface((self.screen.get_width(), self.screen.get_height()))
        # Blit the screen on the Surface
        capture.blit(self.screen, (0, 0))
        # Convert from Surface to Array
        captured = pygame.surfarray.array3d(capture)
        # Return the capture
        return captured         
            
    

Enviroment().run()
'''
EPISODES = 10
save_net_indicator = 1
record_scores = []
record_losses = []
# Initialise pygame
pygame.init()
# Set window
screen = pygame.display.set_mode((405, 410))

enviroment = Enviroment(screen)

for epis in range(1, EPISODES+1):
        
    print("EPISODE: ", epis)
    enviroment.run()
    record_scores.append(enviroment.scores)
    if len(enviroment.losses) == 0:
        pass
    else:
        record_losses.append(np.mean(enviroment.losses))
    

    # Save network conditions
    if (enviroment.scores >= 400) or (epis % 100 == 0):
        # Save the weights
        enviroment.drl_algorithm.q_net.save_weights(f'./records/networks/{save_net_indicator}/')
        # Save network records
        with open("./records/save_network.txt", 'a') as file:
            file.write("Save: {0}, Episode: {1}/{2}, Score: {3}, Epsilon: {4}, Epsilon_action: {5}, Play_time: {6}, Spawn distance: {7}\n".format(save_net_indicator, epis, EPISODES, enviroment.scores, enviroment.drl_algorithm.epsilon, enviroment.drl_algorithm.epsilon_action, enviroment.run_time, enviroment.utils.spawn_eucl_dist))
        save_net_indicator += 1
    # Save record frequency 
    if epis % 100 == 0:        
        # Save metric records
        with open("./records/save_records.txt", 'a') as file:
            file.write("Scores: {0}, Losses: {1}\n".format(record_scores, record_losses))

        tree_priority = pd.DataFrame(enviroment.memory.tree.tree)
        tree_priority.to_csv("./records/tree_idx.csv", index=False, header=None, mode='w')  
                
        data_priority = pd.DataFrame(enviroment.memory.tree.data)
        data_priority.to_csv("./records/data_idx.csv", index=False, header=None, mode='w')

        data_ptr = pd.DataFrame(enviroment.memory.experience_ind)
        data_ptr.to_csv("./records/data_ptr.csv", index=False, header=None, mode='w')
        
        save_net_indicator += 1
        record_scores = []
        record_losses = []
'''



