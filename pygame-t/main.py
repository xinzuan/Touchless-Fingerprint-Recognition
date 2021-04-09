import pygame
import pygame.freetype
from pygame.sprite import Sprite
from pygame.rect import Rect
from enum import Enum
from pygame.sprite import RenderUpdates
from elements import InputBox,UIElement,ImageResult
from state import AppState
from ipwebcam import IPWEBCAM
from user import User
from message import ShowMessage
from preprocess import PreprocessImage

import tkinter
from tkinter import *
from tkinter import filedialog, messagebox
from tkinter.messagebox import *

BLUE = (106, 159, 181)
WHITE = (255, 255, 255)
BGCOLOR =(30,30,30)
PYGAME_WIDTH = 800
PYGAME_HEIGHT =600

BOX_WIDTH_SIZE = 140
BOX_HEIGHT_SIZE = 32

SPACE = 50

MSG = "<No File Selected>"

def prompt_file():
    """Create a Tk file dialog and cleanup when finished"""
    top = tkinter.Tk()
    top.withdraw()  # hide window
    file_name = tkinter.filedialog.askopenfilename(filetypes=[("Image Files", ".png .jpg .jpeg")])
    top.destroy()
    return file_name

def home_screen(screen,font,file_msg):
    start_btn = UIElement(
        center_position=(PYGAME_WIDTH//2, PYGAME_WIDTH//2),
        font_size=30,
        bg_rgb=BLUE,
        text_rgb=WHITE,
        text="Start",
        action=AppState.CAPTURE,
    )

    quit_btn = UIElement(
        center_position=(PYGAME_WIDTH//2, PYGAME_WIDTH//2 + SPACE),
        font_size=30,
        bg_rgb=BLUE,
        text_rgb=WHITE,
        text="Quit",
        action=AppState.QUIT,
    )
    host_label = UIElement(
        center_position=(PYGAME_WIDTH//8 - SPACE, PYGAME_HEIGHT//8-SPACE+BOX_HEIGHT_SIZE//2),
        font_size=20,
        bg_rgb=BGCOLOR,
        text_rgb=WHITE,
        text="Host : ",
        action=None,
        hover= False
    )
    input_host = InputBox(font,PYGAME_WIDTH//8, PYGAME_HEIGHT//8-SPACE, BOX_WIDTH_SIZE, BOX_HEIGHT_SIZE,'192.168.100.6')

    port_label = UIElement(
        center_position=(PYGAME_WIDTH//8-SPACE, PYGAME_HEIGHT//8+SPACE+BOX_HEIGHT_SIZE//2),
        font_size=20,
        bg_rgb=BGCOLOR,
        text_rgb=WHITE,
        text="Port : ",
        action=None,
        hover= False
    )
    input_port = InputBox(font,PYGAME_WIDTH//8, PYGAME_HEIGHT//8+SPACE, BOX_WIDTH_SIZE, BOX_HEIGHT_SIZE,'8080')

    input_boxes = [input_host, input_port]

    upload_btn = UIElement(
        center_position=(PYGAME_WIDTH//8, PYGAME_HEIGHT//3),
        font_size=20,
        bg_rgb=BLUE,
        text_rgb=WHITE,
        text="Upload image",
        action=AppState.UPLOAD,
    
    )
  
    img_file_name = UIElement(
        center_position=(PYGAME_WIDTH//2 ,PYGAME_HEIGHT//3+SPACE),
        font_size=15,
        bg_rgb=BGCOLOR,
        text_rgb=WHITE,
        text=file_msg.text,
        action=None,
        hover= False
    )
    buttons = RenderUpdates(start_btn, quit_btn,host_label,port_label,upload_btn,img_file_name)

    return app_loop(screen, buttons,input_boxes)


def capture_screen(screen, ipwebcam):
    return_btn = UIElement(
        center_position=(140, 570),
        font_size=20,
        bg_rgb=BLUE,
        text_rgb=WHITE,
        text="Return to main menu",
        action=AppState.HOME,
    )
    msg_label = UIElement(
        center_position=(140, 570),
        font_size=20,
        bg_rgb=BLUE,
        text_rgb=WHITE,
        text="Return to main menu",
        action=AppState.HOME,
    )
    buttons = RenderUpdates(return_btn)
    
    return app_loop(screen, buttons,[],ipwebcam)

def result_screen(screen,image_result):
    return_btn = UIElement(
        center_position=(140, 570),
        font_size=20,
        bg_rgb=BLUE,
        text_rgb=WHITE,
        text="Home",
        action=AppState.HOME,
    )

    buttons = RenderUpdates(return_btn)
    
    return app_loop(screen,buttons,[],None,image_result)

def app_loop(screen, buttons,input_boxes=[],ipwebcam = None,results=None):
    """ Handles game loop until an action is return by a button in the
        buttons sprite renderer.
    """
    
    while True:
        mouse_up = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mouse_up = True
            for box in input_boxes:
                box.handle_event(event)

        for box in input_boxes:
            box.update()

        screen.fill(BGCOLOR)
        for box in input_boxes:
            box.draw(screen)
        

        for button in buttons:
            ui_action = button.update(pygame.mouse.get_pos(), mouse_up)
            # print(button.text)
            if ui_action is not None:
                if ui_action == AppState.CAPTURE:
                    temp = []
                    for box in input_boxes:
                        temp.append(box.get_text())
                    if temp[0] and temp[1]:
                        return ui_action,temp
                    else:
                        temp=[]
                        return AppState.HOME,temp
                elif ui_action == AppState.UPLOAD:
                    f = prompt_file()
                    if f:
                        file_msg.set_msg(f)
                        return AppState.UPLOAD,f
                    else:
                        return AppState.HOME,[]
                return ui_action

        buttons.draw(screen)

        if ipwebcam:
            ipwebcam.draw(screen)
            
            if not ipwebcam.check_connection():
                print(ipwebcam.check_connection())
                return AppState.HOME,False,ipwebcam.get_error_msg()
            else:
                ui_action = ipwebcam.connect_ipwebcam()
                
                if ui_action is not None:
                  
                    return ui_action[0],True,ui_action[1]
        if results:
            results.draw(screen)     
        # pygame.display.set_caption(f"Frames: File: {MSG}")
        pygame.display.flip()

def main():
    pygame.init()

    screen = pygame.display.set_mode((PYGAME_WIDTH, PYGAME_HEIGHT))
    font = pygame.font.Font(None, 32)
   
    app_state = AppState.HOME

    global ip_host_port
    global file_msg 
    global result

    file_msg = ShowMessage(MSG)
    

    while True:
        if app_state == AppState.HOME:

            app_state,ip_host_port = home_screen(screen,font,file_msg)
            


        if app_state == AppState.CAPTURE:

            ipwebcam = IPWEBCAM(ip_host_port[0],ip_host_port[1], width=PYGAME_WIDTH//2, height=PYGAME_HEIGHT//2) 
            app_state,is_success,result = capture_screen(screen, ipwebcam)
            if not is_success:
              
                top = tkinter.Tk()
                top.withdraw()  # hide window
                messagebox.showerror("Error",result)
               
                top.destroy()
    
                
                # pass


        if app_state == AppState.UPLOAD:
            
            preprocess = PreprocessImage()
            img_res = preprocess.preprocess_image(file_msg.text)
            image_result = ImageResult(img_res,PYGAME_WIDTH//4,PYGAME_HEIGHT//2)
            app_state = result_screen(screen,image_result)
            file_msg.set_msg(MSG)

        if app_state == AppState.RESULT:

            preprocess = PreprocessImage()
            img_res = preprocess.preprocess_image(result)
            image_result = ImageResult(img_res,PYGAME_WIDTH//4,PYGAME_HEIGHT//2)
            app_state = result_screen(screen,image_result)


        if app_state == AppState.QUIT:
            pygame.quit()
            return


if __name__ == "__main__":
    main()