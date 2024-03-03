import pygame

class Sensor:

    def __init__(self):
        self.position = []        
        self.color = (255,0,0)
        # Bottom line
        self.bottom_line = None
        self.end_pos_bottom_line = [0,200]
        # Left line
        self.left_line = None
        self.end_pos_left_line = [0,0]
        # Upper line
        self.upper_line = None
        self.end_pos_upper_line = [0,0]
        # Right line
        self.right_line = None
        self.end_pos_right_line = [200,0]

    def update_position(self, position):
        self.position = position       

    def lidar(self, screen):
        # Update bottom line position
        self.end_pos_bottom_line[0] = self.position[0]        
        self.bottom_line = pygame.draw.line(screen, self.color, self.position, self.end_pos_bottom_line, 3)
        # Update left line position
        self.end_pos_left_line[1] = self.position[1]        
        self.left_line = pygame.draw.line(screen, self.color, self.position, self.end_pos_left_line, 3)
        # Update upper line position
        self.end_pos_upper_line[0] = self.position[0]        
        self.upper_line = pygame.draw.line(screen, self.color, self.position, self.end_pos_upper_line, 3)
        # Update right line position
        self.end_pos_right_line[1] = self.position[1]        
        self.right_line = pygame.draw.line(screen, self.color, self.position, self.end_pos_right_line, 3)
