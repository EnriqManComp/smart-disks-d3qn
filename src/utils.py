import numpy as np
import pygame

class Utils:

    def __init__(self, obstacles, sensors):
        self.obstacles = obstacles
        self.lidar = sensors
        self.spawn_eucl_dist = 0.0

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
            # x
            # Limit in x => 10 (wall limit) + 20 (robot) + 2 [32, 373]
            x = np.random.randint(low= 20, high=180)
            # Limit in y => 10 (wall limit) + 20 (robot) + 2 [32, 378]
            y = np.random.randint(low= 20, high=180)
            
            # If the spawn overlap the evasor area reset the spawn
            reference_pursuiter = pygame.draw.circle(screen, (0,0,255), (x, y), 8)

            dist = np.sqrt( np.power((evasor_pos[0] - x), 2) +  np.power((evasor_pos[1] - y), 2))

            while reference_pursuiter.colliderect(evasor.robot) or (dist <= 38):
                # New x and y
                x = np.random.randint(low= 20, high=180)
                y = np.random.randint(low= 20, high=180)
                reference_pursuiter = pygame.draw.circle(screen, (0,0,255), (x, y), 8)                
                dist = np.sqrt( np.power((evasor_pos[0] - x), 2) +  np.power((evasor_pos[1] - y), 2))
            
            self.spawn_eucl_dist = np.sqrt( np.power((evasor_pos[0] - x), 2) +  np.power((evasor_pos[1] - y), 2) )
            
        return x, y
    
    def collision(self, pursuiter_rect, evasor_rect, pursuiter_pos, evasor_pos):
        # Collision with the limit walls of the world
        eucl_dist = np.sqrt( np.power((evasor_pos[0] - pursuiter_pos[0]), 2) +  np.power((evasor_pos[1] - pursuiter_pos[1]), 2) )        

        if pursuiter_rect.colliderect(self.obstacles.left_wall):                        
            return "COLLISION"    
        elif pursuiter_rect.colliderect(self.obstacles.top_wall):            
            return "COLLISION"
        elif pursuiter_rect.colliderect(self.obstacles.right_wall):
            return "COLLISION"
        elif pursuiter_rect.colliderect(self.obstacles.bottom_wall):
            return "COLLISION"
        elif eucl_dist <= 38.0:        
            
            if pursuiter_rect.colliderect(evasor_rect):
                return "GOAL-COLLISION-EVASOR"   
            return "GOAL"     
        else:
            return "LIVING PENALTY"
        
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
            
            

        
            