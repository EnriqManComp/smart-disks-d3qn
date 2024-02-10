import pygame

class Pursuiter:

    def __init__(self):
        
        # x, y position
        self.position = []        
        ## GRAPHICS
        self.color = (0, 0, 255)
        self.robot = None    
    
    def spawn(self, surface):
        self.robot = pygame.draw.circle(surface, self.color, (self.position[0], self.position[1]), 20)        
    
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
            if self.position[1] < 15:
                self.position[1] = 15                        
            else:
                self.position[1] -= 5                    
        elif action == "DOWN":
            if self.position[1] > 395:
                self.position[1] = 395
            else:
                self.position[1] += 5
        elif action == "RIGHT":
            if self.position[0] > 390:
                self.position[0] = 390
            else:
                self.position[0] += 5
        elif action == "LEFT":
            if self.position[0] < 30:
                self.position[0] = 30
            else:
                self.position[0] -= 5
        else:
            pass
        
        
        
        
                            
