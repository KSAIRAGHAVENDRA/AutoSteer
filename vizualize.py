import numpy as np
import pandas as pd
import pygame
import glob

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)

preds = pd.read_csv("submissions/final_predictions.csv")
true = pd.read_csv("submissions/CH2_final_evaluation.csv")
filenames = glob.glob("G:/Project/nvidia4/data/test/center/*.jpg")

pygame.init()
size = (640, 320)
pygame.display.set_caption("Data viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
myfont = pygame.font.SysFont("monospace", 15)

for i in range(200,2500):
#for i in range(len(filenames)):
    angle = preds["steering_angle"].iloc[i] # radians
    true_angle = true["angle"].iloc[i] # radians
    if(int(preds["frame_id"].iloc[i].split('\\')[1]) != int(true["frame_id"].iloc[i])):
    	print(int(preds["frame_id"].iloc[i].split('\\')[1])+" "+int(true["frame_id"].iloc[i]))
    
    # add image to screen
    img = pygame.image.load(filenames[i])
    screen.blit(img, (0, 0))
    
    # add text
    pred_txt = myfont.render("Prediction:" + str(round(angle* 57.2958, 3)), 1, (255,255,0)) # angle in degrees
    true_txt = myfont.render("True angle:" + str(round(true_angle* 57.2958, 3)), 1, (255,255,0)) # angle in degrees
    screen.blit(pred_txt, (10, 280))
    screen.blit(true_txt, (10, 300))

    # draw steering wheel
    radius = 50
    pygame.draw.circle(screen, WHITE, [320, 300], radius, 2) 

    # draw cricle for true angle
    x = radius * np.cos(np.pi/2 + true_angle)
    y = radius * np.sin(np.pi/2 + true_angle)
    pygame.draw.circle(screen, WHITE, [320 + int(x), 300 - int(y)], 7)
    
    # draw cricle for predicted angle
    x = radius * np.cos(np.pi/2 + angle)
    y = radius * np.sin(np.pi/2 + angle)
    pygame.draw.circle(screen, BLACK, [320 + int(x), 300 - int(y)], 5) 
    
    #pygame.display.update()
    pygame.display.flip()
    pygame.time.delay(23)
    