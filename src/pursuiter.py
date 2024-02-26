import pygame

class Pursuiter:

    def __init__(self):
        
        # x, y position
        self.position = []        
        ## GRAPHICS
        self.color = (0, 0, 255)
        self.robot = None    
    
    def spawn(self, surface):
        self.robot = pygame.draw.circle(surface, self.color, (self.position[0], self.position[1]), 8)        
    
    def controls(self, action="NO ACTION"):
        '''
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
        '''
        if action == 'UP':
            if self.position[1] < 18:
                self.position[1] = 18                        
            else:
                self.position[1] -= 2
        elif action == "DOWN":
            if self.position[1] > 188:
                self.position[1] = 188
            else:
                self.position[1] += 2
        elif action == "RIGHT":
            if self.position[0] > 188:
                self.position[0] = 188
            else:
                self.position[0] += 2
        elif action == "LEFT":
            if self.position[0] < 18:
                self.position[0] = 18
            else:
                self.position[0] -= 2
        else:
            pass
        
        
        
        
                            
