import sys
import pygame
from ipwebcam import IPWEBCAM

pygame.init()

clock = pygame.time.Clock()
screen = pygame.display.set_mode((800,800))

ipcam = IPWEBCAM('192.168.100.5','8080', width=600, height=600) 
zoom = 0

run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_f:
                ipcam.swap_camera()
            if event.key == pygame.K_b:
                ipcam.swap_camera('off')
            if event.key == pygame.K_o:
                ipcam.overlay("on")
            if event.key == pygame.K_p:
                ipcam.overlay()
            if event.key == pygame.K_l:
                ipcam.led("on")
            if event.key == pygame.K_m:
                ipcam.led()
            if event.key == pygame.K_g:
                # Landscape
                ipcam.set_orientation(0)
            if event.key == pygame.K_h:
                # Portait
                ipcam.set_orientation(1)
            if event.key == pygame.K_0:
                img,roi = ipcam.get_image()
                ipcam.snapshot(roi)
            if event.key == pygame.K_1:
                ipcam.set_resolution("1")
            if event.key == pygame.K_2:
                ipcam.set_resolution("2")
            if event.key == pygame.K_3:
                ipcam.set_resolution("3")
            if event.key == pygame.K_4:
                ipcam.set_resolution("4")
            if event.key == pygame.K_5:
                ipcam.set_resolution("5")
            if event.key == pygame.K_UP:
                zoom += 25
                if zoom > 100:
                    zoom = 100
                ipcam.zoom(zoom)
            if event.key == pygame.K_DOWN:
                zoom -= 25
                if zoom < 0:
                    zoom = 0
                ipcam.zoom(zoom)
            
    screen.blit(ipcam.get_pygame_image(),(0,0))
    pygame.display.flip()
    clock.tick(0)