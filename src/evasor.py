import pygame

class Evasor:

    def __init__(self):
        
        # x, y position
        self.position = []        
        ## GRAPHICS
        self.color = (255, 0, 0)
        self.robot = None    
    
    def spawn(self, surface):
        self.robot = pygame.draw.circle(surface, self.color, (self.position[0], self.position[1]), 8)        
    
    def controls(self, event=None):
        if event is not None:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:                    
                    self.position[0] += 2.0                   
                    
                if event.key == pygame.K_LEFT:
                    self.position[0] -= 2.0
                                        
                if event.key == pygame.K_UP:
                    self.position[1] -= 2.0                    
                     
                if event.key == pygame.K_DOWN:
                    self.position[1] += 2.0                                      
                
        
        
        
                            
