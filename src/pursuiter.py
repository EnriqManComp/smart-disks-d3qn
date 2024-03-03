import pygame

class Pursuiter:

    def __init__(self):        
        # x, y position
        self.position = []        
        ## GRAPHICS
        self.color = (0, 0, 255)
        self.robot = None    
    
    def spawn(self, surface):
        """
        Spawn the pursuiter in the given surface
        
        Input:
            surface: pygame.Surface object

        """
        self.robot = pygame.draw.circle(surface, self.color, (self.position[0], self.position[1]), 8) 
        return       
    
    def controls(self, action="NO ACTION"):       
        """
        Controls the pursuiter movement
        
        Input:
            action: str, action to take
            
        """
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
        elif action == 'DOUBLE-UP':
            if self.position[1] < 18:
                self.position[1] = 18                        
            else:
                self.position[1] -= 4
        elif action == "DOUBLE-DOWN":
            if self.position[1] > 188:
                self.position[1] = 188
            else:
                self.position[1] += 4
        elif action == "DOUBLE-RIGHT":
            if self.position[0] > 188:
                self.position[0] = 188
            else:
                self.position[0] += 4
        elif action == "DOUBLE-LEFT":
            if self.position[0] < 18:
                self.position[0] = 18
            else:
                self.position[0] -= 4
        else:
            pass
        return
        
        
        
        
                            
