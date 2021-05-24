import pygame
import pygame.freetype
from pygame.sprite import Sprite
from pygame.rect import Rect
from enum import Enum
from pygame.sprite import RenderUpdates
from elements import InputBox,UIElement,ImageResult,LoadingBar
from state import AppState,PATH
from ipwebcam import IPWEBCAM
from user import User
from message import ShowMessage
from preprocess import PreprocessImage

import tkinter
from tkinter import *
from tkinter import filedialog, messagebox,simpledialog
from tkinter.messagebox import *

import itertools
import threading
import time
import sys
from queue import Queue
import os

BLUE = (106, 159, 181)
WHITE = (255, 255, 255)
BGCOLOR =(30,30,30)
PYGAME_WIDTH = 800
PYGAME_HEIGHT =600

BOX_WIDTH_SIZE = 140
BOX_HEIGHT_SIZE = 32

SPACE = 50

barPos      = (120, 360)
barSize     = (200, 20)
borderColor = (0, 0, 0)
barColor    = (0, 128, 0)



def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rloading ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!     ')

def prompt_file():
    """Create a Tk file dialog and cleanup when finished"""

    top = tkinter.Tk()
    top.withdraw()  # hide window
    file_name = tkinter.filedialog.askopenfilename(initialdir=PATH.SAVE_PATH_RAW_IMAGE.value,filetypes=[("Image Files", ".png .jpg .jpeg")])
    top.destroy()
    return file_name

def home_screen(screen,font):
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
  
    
    buttons = RenderUpdates(start_btn, quit_btn,host_label,port_label,upload_btn)
    # labels = RenderUpdates(img_file_name)
    
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

def result_screen(screen,image_result=[],progressing=True):
    

    if not progressing:
        return_btn = UIElement(
            center_position=(140, 570),
            font_size=20,
            bg_rgb=BLUE,
            text_rgb=WHITE,
            text="Home",
            action=AppState.HOME,
        )
        buttons = RenderUpdates(return_btn)
        return app_loop(screen=screen,buttons=buttons,results=image_result)
    else:
        loading = LoadingBar(progress=1)
        print('a')
        return app_loop(screen=screen,loading=loading)

    
    
    
# def loading_screen(screen,a):
#     # return_btn = UIElement(
#     #         center_position=(140, 570),
#     #         font_size=20,
#     #         bg_rgb=BLUE,
#     #         text_rgb=WHITE,
#     #         text="Home",
#     #         action=AppState.HOME,
#     #     )

#     # buttons = RenderUpdates(return_btn)
#     loading = LoadingBar(progress=a)
#     # pygame.draw.rect(screen, borderC, (*pos, *size), 1)
#     # innerPos  = (pos[0]+3, pos[1]+3)
#     # innerSize = ((size[0]-6) * progress, size[1]-6)
#     # pygame.draw.rect(screen, barC, (*innerPos, *innerSize))
#     return app_loop(screen=screen,loading=loading)

def app_loop(screen, buttons=[],input_boxes=[],ipwebcam = None,origins=None,results=None,loading=None):
    """ Handles game loop until an action is return by a button in the
        buttons sprite renderer.
    """
    

    text = 'No file selected'
    is_upload =False
    
    while True:
        mouse_up = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                mouse_up = True
            for box in input_boxes:
                box.handle_event(event)

        # for label in labels:
        #     label.draw(screen)
        
        for box in input_boxes:
            box.update()

        screen.fill(BGCOLOR)
        for box in input_boxes:
            box.draw(screen)
        
      
        

        
        # DrawBar(screen,barPos, barSize, borderColor, barColor, a/max_a)
        for button in buttons:
            ui_action = button.update(pygame.mouse.get_pos(), mouse_up)
           
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
                    # if text:
                    #     return AppState.UPLOAD,f
                    if f:
                        
                        # file_msg.set_msg(f)
                        text = f
                 
                        # updatetext(screen,text,font2)
                        #DrawBar(screen,barPos, barSize, borderColor, barColor, a/max_a)
                        is_upload = True
                       
                        continue
                        
                        # return AppState.UPLOAD,f
                    else:
                        return AppState.HOME,[]
                    
                return ui_action
        if buttons:
            buttons.draw(screen)
        updatetext(screen,text,font2)
        if is_upload: 
            # print(text)
            return AppState.UPLOAD,text
        
        if ipwebcam:
            ipwebcam.draw(screen)
            
            if not ipwebcam.check_connection():
                
                return AppState.HOME,False,ipwebcam.get_error_msg()
            else:
                ui_action = ipwebcam.connect_ipwebcam()
                
                if ui_action is not None:
                  
                    return ui_action[0],True,ui_action[1]
        if results:
            for i in results:
                i.draw(screen) 
        if loading:
            loading.draw(screen) 
            
            loading.update()
            time.sleep(0.5)
            
            
        #loading.update()    
        # pygame.display.set_caption(f"Frames: File: {MSG}")
        pygame.display.flip()

def run_preprocess(image):
    # img_res,fsrcnn_res,edsr_res,esrgan_res = preprocess.preprocess_image(image)
    return preprocess.preprocess_image(image)


def main():
    pygame.init()

    screen = pygame.display.set_mode((PYGAME_WIDTH, PYGAME_HEIGHT))
    global font2 
    font = pygame.font.Font(None, 32)
    font2 = pygame.font.Font(None, 20)
    app_state = AppState.HOME

    global ip_host_port
    
    global result
    on_process = True
    global preprocess
    preprocess = PreprocessImage()
    

 
    a =0

    while True:
        # print(app_state)
        # print(text)
        if app_state == AppState.HOME:

            app_state,ip_host_port = home_screen(screen,font)

                
        

        if app_state == AppState.CAPTURE:

            ipwebcam = IPWEBCAM(ip_host_port[0],ip_host_port[1], width=PYGAME_WIDTH, height=PYGAME_HEIGHT) 
            
            
            app_state,is_success,result = capture_screen(screen, ipwebcam)
            if not is_success:
              
                top = tkinter.Tk()
                top.withdraw()  # hide window
                messagebox.showerror("Error",result)
               
                top.destroy()
                pass
    
            # print(app_state)
            # print(result)  
                # pass


        if app_state == AppState.UPLOAD:
            # app_state,ip_host_port = result(screen,font)
            # print('Loading...')
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     print('aa')
            #     future = executor.submit(run_preprocess, ip_host_port)

            #     img_res,fsrcnn_res,edsr_res,esrgan_res = future.result()
            #     print(img_res)
            
            que = Queue()

            t = threading.Thread(target=lambda q, arg1: q.put(run_preprocess(arg1)), args=(que, ip_host_port))
            
            t.start()
            max_a = 100
            a = 0
            
            # loading = LoadingBar(progress=1)
            # loading.draw(screen)
            # app_state = result_screen(screen,progressing=True,loading=loading)
            # while t.isAlive():
                
            #     # loading.update()
            #     print('aa')
            #     app_state = result_screen(screen,progressing=t.isAlive())
               
            #     if not t.isAlive():
            #         break
            #     # print('a')
            # # else:
            # #     app_state = result_screen(screen,progressing=False)
            # #     pass
            # app_state = result_screen(screen,progressing=True)
            t.join()
            
            img_res,fsrcnn_res,edsr_res,esrgan_res = que.get()
            
            
                
            # print('a')
            
            
            if img_res:
                
                probe = preprocess.get_matcher(img_res)
                if not probe:
                    top = tkinter.Tk()
                    top.withdraw()
                    messagebox.showerror(title='Error', message='Cannot connect to server')
                    top.destroy()
                    app_state = AppState.HOME
                else:
                    if probe['data']['name'] != '-':
                        top = tkinter.Tk()
                        top.withdraw()  # hide window
                        messagebox.showinfo(title='Verification', message='Hello, ' + probe['data']['name'])
                        # messagebox.showinfo(title='Verification', message='Hasil penilaian : {:.2f}'.format(probe['data']['score']))
                        top.destroy()
                        images =[]
                        image_origin = ImageResult(ip_host_port,PYGAME_WIDTH//5,PYGAME_HEIGHT//3,x=250,y=50)
                        image_result = ImageResult(img_res,PYGAME_WIDTH//5,PYGAME_HEIGHT//3,x = 450,y=50)
                        # fsrcnn_res = ImageResult(img_res,PYGAME_WIDTH//5,PYGAME_HEIGHT//3,x = 50,y=300)
                        # edsr_res = ImageResult(img_res,PYGAME_WIDTH//5,PYGAME_HEIGHT//3,x = 350,y=300)
                        # esrgan_res = ImageResult(img_res,PYGAME_WIDTH//5,PYGAME_HEIGHT//3,x = 450,y = 300)
                        # images =[image_origin,image_result,fsrcnn_res,edsr_res,esrgan_res]
                        images =[image_origin,image_result]
                        app_state = result_screen(screen,images,progressing=False)
                        
                    else:
                        top = tkinter.Tk()
                        top.withdraw()
                        answer = messagebox.askyesno(title='Unauthorized',message='Want to retry?')
                        top.destroy()
                        if answer:
                            app_state = AppState.HOME
                        else:
                            top = tkinter.Tk()
                            top.withdraw()
                            answer = messagebox.askyesno(title='Confirmation',message='Create new identity?')
                            top.destroy()
                            if answer:
                                top = tkinter.Tk()
                                top.withdraw()
                                user_name = simpledialog.askstring(title="Test",
                                    prompt="What's your name?:")
                                top.destroy()
                                success_save = preprocess.add_new_image(img_res,user_name)
                                
                                if success_save:
                                    top = tkinter.Tk()
                                    top.withdraw()
                                    messagebox.showinfo(title='Success', message='Data created')
                                    top.destroy()
                                    app_state = AppState.HOME
                                else:
                                    top = tkinter.Tk()
                                    top.withdraw()
                                    messagebox.showerror(title='Error', message='Failed to create identity')
                                    top.destroy()
                                    app_state = AppState.HOME
                            else:
                                app_state = AppState.HOME
                                # check it out
                                # print("Hello", USER_INP)

                                # app_state = AppState.HOME

            


            
            

        # if app_state == AppState.RESULT:

            
            
        if app_state == AppState.QUIT:
            pygame.quit()
            return

def updatetext (screen,value,font):
    
    mixLED = font.render(value, 1, (255,255,255))
    
    screen.blit(mixLED, (PYGAME_WIDTH//4, PYGAME_HEIGHT//3))
def DrawBar(screen,pos, size, borderC, barC, progress):

    pygame.draw.rect(screen, borderC, (*pos, *size), 1)
    innerPos  = (pos[0]+3, pos[1]+3)
    innerSize = ((size[0]-6) * progress, size[1]-6)
    pygame.draw.rect(screen, barC, (*innerPos, *innerSize))

if __name__ == "__main__":
    main()