import numpy as np
import pygame

class Utils:

    def __init__(self, obstacles, sensors):
        self.obstacles = obstacles
        self.lidar = sensors
        self.spawn_eucl_dist = 0.0
        
        ### REWARDS
        self.REWARDS = {
            "COLLISION": -1000,            
            "GOAL": +1000            
        } 
        self.danger_zone = 25.0
        self.target_zone = 45.0


    def eucl_distance(self, x1, y1, x2, y2):
        return np.sqrt(np.power((x2-x1), 2) + np.power((y2-y1), 2))

    def random_spawn(self, evasor_pos=None, screen= None, evasor=None, evasor_spawn=False):
        ##### Random coordinates
        if evasor_spawn:        
            print("SPAWING EVASOR...")
            # x
            # Limit in x => 10 (wall limit) + 20 (robot) + 2 [32, 373]
            x = np.random.uniform(low= 20, high= 180)
            # Limit in y => 10 (wall limit) + 20 (robot) + 2 [35, 378]
            y = np.random.uniform(low= 20, high= 180)            
        else:           
            print("SPAWING PURSUITER...")
            
            ##### Curriculum Learning
            # Complex task
            # x
            # Limit in x => 10 (wall limit) + 20 (robot) + 2 [32, 373]
            #x = np.random.randint(low= 20, high=180)
            # Limit in y => 10 (wall limit) + 20 (robot) + 2 [32, 378]
            #y = np.random.randint(low= 20, high=180)

            # Easy task           
            # Spawn the pursuiter within the evasor area of 100 pixels
            # Give 10 pixels of threshold to avoid spawn in the target zone
            low_limit_x = evasor_pos[0] + self.target_zone + 10
            low_limit_y = evasor_pos[1] + self.target_zone + 10
            high_limit_x = evasor_pos[0] + 100
            high_limit_y = evasor_pos[1] + 100
            x = np.random.randint(low= low_limit_x , high= high_limit_x)
            y = np.random.randint(low= low_limit_y, high= high_limit_y)
            # If the spawn overlap the evasor area reset the spawn
            reference_pursuiter = pygame.draw.circle(screen, (0,0,255), (x, y), 8)            

            dist = self.eucl_distance(x, y, evasor_pos[0], evasor_pos[1])
            
            while reference_pursuiter.colliderect(evasor.robot) or (dist <= self.target_zone):
                # New x and y
                x = np.random.randint(low= low_limit_x, high= high_limit_x)
                y = np.random.randint(low= low_limit_y, high= high_limit_y)
                reference_pursuiter = pygame.draw.circle(screen, (0,0,255), (x, y), 8)                
                dist = self.eucl_distance(x, y, evasor_pos[0], evasor_pos[1])
            
            self.spawn_eucl_dist = self.eucl_distance(x, y, evasor_pos[0], evasor_pos[1])
            
        return x, y
    
    def get_reward(self, pursuiter_rect, evasor_rect, pursuiter_pos, evasor_pos):
        # Collision with the limit walls of the world        
        dist_p_e = self.eucl_distance(pursuiter_pos[0], pursuiter_pos[1], evasor_pos[0], evasor_pos[1])
        fc = 0.0
        fp = 0.0
        done = False
        # Check left wall collision
        if pursuiter_rect.colliderect(self.obstacles.left_wall):                        
            fc = self.REWARDS["COLLISION"]
            done = True
        # Check upper wall collision
        elif pursuiter_rect.colliderect(self.obstacles.top_wall):            
            fc = self.REWARDS["COLLISION"]
            done = True
        # Check right wall collision
        elif pursuiter_rect.colliderect(self.obstacles.right_wall):
            fc = self.REWARDS["COLLISION"]
            done = True
        # Check bottom wall collision
        elif pursuiter_rect.colliderect(self.obstacles.bottom_wall):
            fc = self.REWARDS["COLLISION"]
            done = True
        # Check collision with the evasor
        elif pursuiter_rect.colliderect(evasor_rect):
            fc = self.REWARDS["COLLISION"]
            done = True
        # Check if the pursuiter is in the danger zone in the left and the distance to the evasor is not in the target zone (left wall danger zone)
        elif (self.dist_to_left <= self.danger_zone and dist_p_e > self.target_zone):
            fc = self.danger_zone_rewards(self.dist_to_left)
        # Check if the pursuiter is in the danger zone in the upper and the distance to the evasor is not in the target zone (upper wall danger zone)
        elif (self.dist_to_upper <= self.danger_zone and dist_p_e > self.target_zone):
            fc = self.danger_zone_rewards(self.dist_to_upper)
        # Check if the pursuiter is in the danger zone in the right and the distance to the evasor is not in the target zone (right wall danger zone)
        elif (self.dist_to_right <= self.danger_zone and dist_p_e > self.target_zone):
            fc = self.danger_zone_rewards(self.dist_to_right)
        # Check if the pursuiter is in the danger zone in the bottom and the distance to the evasor is not in the target zone (bottom wall danger zone)
        elif (self.dist_to_bottom <= self.danger_zone and dist_p_e > self.target_zone):
            fc = self.danger_zone_rewards(self.dist_to_bottom)
        # If any case is true, the reward fc is 0.0
        else:
            fc = 0.0       
        # Check if the pursuiter is in the target zone
        if dist_p_e <= self.target_zone:          
            fp = self.REWARDS["GOAL"]
            # Check if the pursuiter is in the danger zone respect to the evasor
            if dist_p_e <= self.danger_zone:
                # The 2 factor is added because fc return 0.0 when the pursuiter is in the danger zone and dist_p_e is in target zone
                fp = 2 * self.danger_zone_rewards(dist_p_e)                
        else:
            # Living positive penalty
            fp = (10 / dist_p_e)
        # Total Reward
        return (fc + fp), done

    def danger_zone_rewards(self, eucl_dist):
        rate = (self.danger_zone - eucl_dist) / (self.danger_zone + eucl_dist)
        return (self.REWARDS["COLLISION"] * rate)
        
    def lidar_observations(self, pos_x, pos_y, evasor):                   
        # Bottom limit collision                
        if self.obstacles.bottom_wall.colliderect(self.lidar.bottom_line):
            if evasor.robot.colliderect(self.lidar.bottom_line):
                self.dist_to_bottom = np.abs(int(pos_y - evasor.position[1]))               
            else:
                self.dist_to_bottom = np.abs(pos_y - 200 + 10)            
        # Left limit collision
        if self.obstacles.left_wall.colliderect(self.lidar.left_line):
            if evasor.robot.colliderect(self.lidar.left_line):
                self.dist_to_left = np.abs(int(pos_x - evasor.position[0]))
            else:
                self.dist_to_left = pos_x + 10 
        # Upper limit collision
        if self.obstacles.top_wall.colliderect(self.lidar.upper_line):
            if evasor.robot.colliderect(self.lidar.upper_line):
                self.dist_to_upper = np.abs(int(pos_y - evasor.position[1]))
            else:
                self.dist_to_upper = pos_y + 10
        # Right limit collision
        if self.obstacles.right_wall.colliderect(self.lidar.right_line):
            if evasor.robot.colliderect(self.lidar.right_line):
                self.dist_to_right = np.abs(int(pos_x - evasor.position[0]))
            else:
                self.dist_to_right = np.abs(pos_x  - 200 + 10)
        return [self.dist_to_left, self.dist_to_upper, self.dist_to_right, self.dist_to_bottom]            
            
            

        
            