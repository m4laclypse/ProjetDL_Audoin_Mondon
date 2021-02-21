# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 13:58:18 2021

@author: loic9
"""

import pygame
import sys
import math
import pickle
from random import randrange as randH

from pygame.constants import KEYDOWN, K_ESCAPE, QUIT
from pygame.locals import *

# by John Mathison

# Ground Elevation (pixels)
groundLevel = 400

# Global colors
birdColor = pygame.Color('#222222')
backgroundColor = pygame.Color('#abcdef')
groundColor = pygame.Color('#993333')
fontColor = pygame.Color('#FFFFFF')


# Class for pipe obstacles
class Pipes:

    height = 0
    width = 60
    gap = 250  # 150
    pos = 600
    replaced = False
    scored = False
    first = False

    # Randomize pipe location
    def __init__(self):
        self.height = randH(210, groundLevel - 10)

    # Moves the pipes along the ground, checks if they're off the screen
    def move(self, movement):
        self.pos += movement
        if (self.pos + self.width < 0):
            return False  # Return false if we moved off the screen
        return True

    # Handles drawing the pipes to the screen
    def draw(self, surface):
        pygame.draw.rect(surface, groundColor, (self.pos, self.height, self.width, groundLevel - self.height))
        pygame.draw.rect(surface, groundColor, (self.pos, 0, self.width, self.height - self.gap))


# Class for the player
class Bird:

    pos = (0, 0)
    radius = 20

    def __init__(self, newPos):
        self.pos = newPos

    # Handles drawing the bird to the screen
    def draw(self, surface):
        intPos = (int(math.floor(self.pos[0])), int(math.floor(self.pos[1])))

        pygame.draw.circle(surface, birdColor, intPos, self.radius)

    # Attempt to move the bird, make sure we aren't hitting the ground
    def move(self, movement):
        posX, posY = self.pos
        movX, movY = movement

        if((posY + movY + self.radius) < groundLevel):
            self.pos = (posX + movX, posY + movY)
            return True  # Return if we successfuly moved
        self.pos = (posX, groundLevel - self.radius)
        return False

    # Test for collision with the given pipe
    def collision(self, pipe):
        posX, posY = self.pos
        collideWidth = (pipe.pos < posX + self.radius and posX - self.radius < pipe.pos + pipe.width)
        collideTop = (pipe.height - pipe.gap > posY - self.radius)
        collideBottom = (posY + self.radius > pipe.height)
        if (collideWidth and (collideTop or collideBottom)):
            return True
        return False


# Main game loop
class FlappyBird:
    def __init__(self, graphique=True, FPS=30, save=None, quickstart=False):
        # Setting up initial values
        pygame.init()
        if graphique:
            self.windowObj = pygame.display.set_mode((640, 480))
            self.fpsTimer = pygame.time.Clock()

        self.bird = Bird((640 / 4, 480 / 2))

        if quickstart:
            pipe1 = Pipes()
            pipe1.pos = 320
            pipe1.height = 320
            pipe1.First = True
            pipe1.replaced = True
            pipe2 = Pipes()
            pipe2.pos = 600
            pipe2.height = 320
            self.pipes = [pipe1, pipe2]
        else:
            self.pipes = [Pipes()]

        self.gravity = 2
        self.velocity = 0
        self.score = 0
        self.highScore = 0
        self.graphique = graphique
        self.FPS = FPS
        self.gap = 150
        self.save = save
        self.data=[]
        self.quickstart = quickstart
        self.fontObj = pygame.font.Font(None, 16)

    def getState(self):
        if len(self.pipes) == 1:
            premierPipeHeight = (self.pipes[0].height - 210) / 390
            premierPipePos = self.pipes[0].pos / 600

            deuxiemePipeHeight = 0.
            deuxiemePipePos = 0.

        else:
            compteur = 0
            while self.pipes[compteur].pos <= 160:
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
        self.data = self.data[:-10]
        if (self.score > self.highScore):
            self.highScore = self.score
        self.score = 0
        self.velocity = 0

        if self.quickstart:
            pipe1 = Pipes()
            pipe1.pos = 320
            pipe1.height = 320
            pipe1.First = True
            pipe1.replaced = True
            pipe2 = Pipes()
            pipe2.pos = 600
            pipe2.height = 320
            self.pipes = [pipe1, pipe2]
        else:
            self.pipes = [Pipes()]
        self.bird.pos = ((640 / 4, 480 / 2))

    def pause(self):
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.exit()
                elif event.type == KEYDOWN:
                    if (event.key == K_ESCAPE):
                        return

    def exit(self):
        if self.save is not None:
            file = open(self.save, "bw+")
            pickle.dump(self.data, file)
        pygame.quit()
        sys.exit()
    
    def getPlayerHistory(self):
        return self.data

    def nextFrame(self, manual=False, entry=None):
        lossValue = 0 
        if self.graphique:
            self.windowObj.fill(backgroundColor)

        hasJumped = False
        if not manual:
            # Check for events
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    hasJumped = True
                    if (event.key == K_ESCAPE):
                        self.pause()
                    # If the player hits a key, set velocity upward
                    self.velocity = -20
        
        if hasJumped:
            self.data.append((normalize(self.getState()), "jump"))
        else:
            self.data.append((normalize(self.getState()), "stay"))

        if len(self.data) % 100 == 0:
            print("On a ", len(self.data), " points de données")
        else:
            if entry == "quit":
                pygame.quit()
                sys.exit()
            elif entry == "jump":
                self.velocity = -20

        # Add acceleration from gravity
        self.velocity += self.gravity

        if (not self.bird.move((0, self.velocity))):
            lossValue = 1
            self.resetGame()
            self.velocity = 0
        for pipe in self.pipes:
            if not pipe.replaced and pipe.pos < 640 / 2 and not pipe.first:
                self.pipes[len(self.pipes):] = [Pipes()]
                pipe.replaced = True
            if self.graphique:
                pipe.draw(self.windowObj)
            if (self.bird.collision(pipe)):
                if self.bird.pos[1] < pipe.height - pipe.gap:
                    lossValue = abs(self.bird.pos[1] - pipe.height + pipe.gap) / 480
                else:
                    lossValue = abs(self.bird.pos[1] - pipe.height) / 480
                if self.graphique:
                    self.windowObj.fill(pygame.Color('#230056'))
                self.resetGame()
            if (not pipe.scored and pipe.pos + pipe.width < self.bird.pos[0]):
                self.score += 1
                pipe.scored = True
            if(not pipe.move(-10)):
                del pipe

        # Draw stuff
        scoreSurface = self.fontObj.render('Score: ' + str(self.score) + ' High: ' + str(self.highScore),
                                           False, fontColor)
        scoreRect = scoreSurface.get_rect()
        if self.graphique:
            scoreRect.topleft = (self.windowObj.get_height() / 2, 10)
            self.windowObj.blit(scoreSurface, scoreRect)
            pygame.draw.rect(self.windowObj, groundColor, (0, groundLevel, self.windowObj.get_width(),
                                                        self.windowObj.get_height()))

            self.bird.draw(self.windowObj)

        if self.graphique:
            pygame.display.update()
            self.fpsTimer.tick(self.FPS)
        return lossValue


def normalize(getstate):
    Lmean = [5.870e-01, 2.158e-01, 2.890e-02, 2.235e-01, 4.622e-01, -4.000e-04]
    Lstd = [0.1286, 0.1291, 0.13, 0.1465, 0.2003, 0.388 ]
    if getstate[2] >= 0.45 :
        getstate[2] = 0.0
    for i in range(6):
        getstate[i] = (getstate[i]-Lmean[i])/Lstd[i]
    if getstate[4] <= -1.75 :
        getstate[4] = 0.0
    return getstate


if __name__ == "__main__":
    flappy = FlappyBird(save="entryDataDepart.pickle.dat")
    while True:
        flappy.nextFrame()
