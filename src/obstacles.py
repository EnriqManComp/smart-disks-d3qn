import pygame

class Obstacles:

    def __init__(self, screen):
        self.screen = screen
        self.wall_color = (255, 200, 0)
        # Left wall
        self.left_wall = None
        # x, y, width, height
        self.left_wall_desc = (0, 0, 10, self.screen.get_height())
        # Top wall
        self.top_wall = None
        self.top_wall_desc = (0, 0, self.screen.get_width(), 10)
        # Right wall
        self.right_wall = None
        self.right_wall_desc = (self.screen.get_width() - 10, 0, 10, self.screen.get_height())
        # Bottom wall
        self.bottom_wall = None
        self.bottom_wall_desc = (0, self.screen.get_height() - 10, self.screen.get_width(), 10)
        

    def render_walls(self):
        """
        
        Render the walls of the world

        """
        #### Limits of the world
        # Render left wall
        self.left_wall = pygame.draw.rect(self.screen, self.wall_color, self.left_wall_desc)
        # Render top wall
        self.top_wall = pygame.draw.rect(self.screen, self.wall_color, self.top_wall_desc)
        # Render Right wall
        self.right_wall = pygame.draw.rect(self.screen, self.wall_color, self.right_wall_desc)
        # Render Bottom wall
        self.bottom_wall = pygame.draw.rect(self.screen, self.wall_color, self.bottom_wall_desc)


    
        

