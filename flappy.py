# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 13:58:18 2021

@author: loic9
"""

import pygame, sys, math
from random import randrange as randH
from pygame.locals import *

# Flappy python? by John Mathison
# Checking to see how long it takes to make flappy bird

# Start pygame
pygame.init()


import pygame, sys, math
from random import randrange as randH
from pygame.locals import *

# Set up resolution
windowObj = pygame.display.set_mode( ( 640, 480) )
fpsTimer = pygame.time.Clock()

# Ground Elevation (pixels)
groundLevel = 400

# Global colors
birdColor = pygame.Color('#222222')
backgroundColor = pygame.Color('#abcdef')
groundColor = pygame.Color('#993333')
fontColor = pygame.Color('#FFFFFF')

# fontObj = pygame.font.Font('OpenSans-Bold.ttf', 16)
fontObj = pygame.font.Font(None, 16)
   

# Class for pipe obstacles
class Pipes:

    height = 0
    width = 60
    gap = 150
    pos = 600;
    replaced = False
    scored = False

    # Randomize pipe location
    def __init__(self):
        self.height = randH(210, groundLevel - 10)

    # Moves the pipes along the ground, checks if they're off the screen
    def move(self, movement):
        self.pos += movement
        if( self.pos + self.width < 0 ):
            return False #Return false if we moved off the screen 
        return True

    # Handles drawing the pipes to the screen
    def draw(self, surface):
        pygame.draw.rect( surface, groundColor, (self.pos, self.height, self.width, groundLevel - self.height))
        pygame.draw.rect( surface, groundColor, (self.pos, 0, self.width, self.height - self.gap))

# Class for the player 
class Bird:
    
    pos = (0,0)
    radius = 20
    
    def __init__(self, newPos):
        self.pos = newPos

    # Handles drawing the bird to the screen
    def draw(self, surface):
        intPos = ( int(math.floor(self.pos[0])), int(math.floor(self.pos[1])) )
        
        pygame.draw.circle(surface, birdColor, intPos, self.radius)

    # Attempt to move the bird, make sure we aren't hitting the ground
    def move(self, movement):
        posX, posY = self.pos
        movX, movY = movement

        if( (posY + movY + self.radius) < groundLevel ):
            self.pos = (posX + movX, posY + movY)
            return True #Return if we successfuly moved
        self.pos = (posX, groundLevel - self.radius)
        return False

    # Test for collision with the given pipe
    def collision(self, pipe):
        posX, posY = self.pos
        collideWidth = ( pipe.pos < posX + self.radius and posX - self.radius < pipe.pos + pipe.width)
        collideTop = ( pipe.height - pipe.gap > posY - self.radius )
        collideBottom = ( posY + self.radius > pipe.height )
        if (  collideWidth and ( collideTop or collideBottom)):
            return True
        return False


# Main game loop
class FlappyBird:
    def __init__(self, graphique=True, FPS=30):
        # Setting up initial values
        self.bird = Bird((windowObj.get_width() / 4 , windowObj.get_height() / 2))
        self.pipes = [ Pipes()]
        self.gravity = 2
        self.velocity = 0
        self.score = 0
        self.highScore = 0
        self.graphique = graphique
        self.FPS = FPS
    
    def getState(self):
        if len(self.pipes) == 1:
            premierPipeHeight = (self.pipes[0].height - 210) / 390
            premierPipePos = self.pipes[0].pos / 600
            
            deuxiemePipeHeight = 1
            deuxiemePipePos = 1

        else:
            compteur = 0
            while self.pipes[compteur].pos <= 160 :
                compteur += 1
            premierPipeHeight = (self.pipes[compteur - 1].height - 210) / 390
            premierPipePos = self.pipes[compteur - 1].pos / 600

            deuxiemePipeHeight = (self.pipes[compteur].height - 210) / 390
            deuxiemePipePos = self.pipes[compteur].pos / 600

        return [self.bird.pos[1] / 400, premierPipeHeight, premierPipePos, deuxiemePipeHeight,
                deuxiemePipePos, self.velocity / 30]

    def getScore(self):
        return self.score
        
    # Called to reset the game when you lose
    def resetGame(self):
        if ( self.score > self.highScore ):
            self.highScore = self.score
        self.score = 0
        self.velocity = 0
        self.pipes = [ Pipes() ]
        self.bird.pos = ( (windowObj.get_width() / 4 , windowObj.get_height() / 2) )

    def pause(self):
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    if ( event.key == K_ESCAPE):
                        return
    
    def nextFrame(self, manual=False, entry=None):
        hasNotLost = True
        windowObj.fill(backgroundColor)

        if not manual:
            # Check for events
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    if ( event.key == K_ESCAPE):
                        self.pause()
                    # If the player hits a key, set velocity upward
                    self.velocity = -20
        else:
            if entry == "quit":
                pygame.quit()
                sys.exit()
            elif entry == "jump":
                self.velocity = -20

        # Add acceleration from gravity
        self.velocity += self.gravity

        if (not self.bird.move((0, self.velocity))):
            hasNotLost = False
            self.resetGame()
            self.velocity = 0
        for pipe in self.pipes:
            if not pipe.replaced and pipe.pos < windowObj.get_width() / 2  :
                self.pipes[len(self.pipes):] = [Pipes()]
                pipe.replaced = True
            pipe.draw(windowObj)
            if (self.bird.collision(pipe)):
                windowObj.fill(pygame.Color('#230056'))
                self.resetGame()
            if ( not pipe.scored and pipe.pos + pipe.width < self.bird.pos[0] ):
                self.score += 1
                pipe.scored = True
            if( not pipe.move(-10)):
                del pipe

        # Draw stuff
        scoreSurface = fontObj.render( 'Score: ' + str(self.score) + ' High: ' + str(self.highScore), False, fontColor)
        scoreRect = scoreSurface.get_rect()
        scoreRect.topleft = (windowObj.get_height() / 2 , 10)
        windowObj.blit(scoreSurface, scoreRect)
        pygame.draw.rect(windowObj, groundColor, (0, groundLevel, windowObj.get_width(), windowObj.get_height()) )

        self.bird.draw(windowObj)

        if self.graphique:
            pygame.display.update()
        fpsTimer.tick(self.FPS)
        return hasNotLost
 

if __name__ == "__main__":
    flappy = FlappyBird()
    while True:
        flappy.nextFrame()
