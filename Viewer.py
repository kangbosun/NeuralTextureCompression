

#simple png viewer
import sys
import os
import pygame
from pygame.locals import *
from pygame.color import THECOLORS

def main():
    # Initialize Pygame
    pygame.init()
    # Set the window size
    screen = pygame.display.set_mode((1920, 1080), RESIZABLE)
    # Set the window title
    pygame.display.set_caption('Simple Image Viewer')
    # Load the image
    image = pygame.image.load('wood.png')
    # Display the image on the screen proper size
    screen.blit(pygame.transform.scale(image, (1920, 1080)), (0, 0))

    # Update the display
    pygame.display.flip()
    # Wait for the window to be closed
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                return
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return
            if event.type == VIDEORESIZE:
                screen = pygame.display.set_mode(event.dict['size'], RESIZABLE)
                screen.blit(pygame.transform.scale(image, event.dict['size']), (0, 0))
                pygame.display.flip()
                
if __name__ == '__main__':
    main()

