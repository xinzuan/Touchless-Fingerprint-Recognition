import urllib
import cv2
import time
import numpy as np
import pygame
import requests as r
import os
from state import AppState
from preprocess import PreprocessImage
from state import PATH

class IPWEBCAM(object):
    def __init__(self,host='192.168.100.3',port='8080', width=400, height=400):
        self.url = "http://"+host+":"+port
        self.width = width
        self.height = height
        self.img_counter =0
        self.x = width //2
        self.y = height //2
        self.upper_left = (self.x-100, self.y-100)
        self.bottom_right = (self.x+300, self.y+500)
        self.connect = False
        self.zoom_num = 0
        self.msg =''
        self.temp={}
        
       
        
    def get_image(self):
        # Get our image from the phone
        try:
            imgResp = urllib.request.urlopen(self.url + '/shot.jpg')
            
            self.connect = True
            # Convert our image to a numpy array so that we can work with it
            imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)

            # Convert our image again but this time to opencv format
            img = cv2.imdecode(imgNp,-1)
            # print(self.x)
            # print(self.y)
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            
            # self.roi=img[ self.y-100:self.y+500,self.x-100:self.x+300]

            
            # cv2.rectangle(img,self.upper_left,self.bottom_right,(255,0,0),2)
            # hsv = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)

            # return True,img,self.roi

            return True,img
        except Exception as e:
            self.msg = e
            print(e)
            return False,e


    def get_image_string(self,img):
        # return the image as a string, but also give out its shape(width,height) and color_space
        return (img.tostring(), img.shape[1::-1], 'RGB')

    def get_pygame_image(self):
        # Get the image
        is_img,res = self.get_image()
        if is_img:
             
            not_blur,value = self.check_blurry(res)
            
            if not_blur :
                # print(value)
                img_name = self.snapshot(res,value)
                #return True,img_name
        
                
                
            # split our image color_space into blue, green, red components
            b,g,r = cv2.split(res)

            # compose our image back but this time as red, green and blue
            res = cv2.merge([r,g,b])

            # get our image in string format and also the size and color_space for pygame to Use
            res,shape,color_space = self.get_image_string(res)

            # create the pygame image from the string, size and color space
            res = pygame.image.frombuffer(res,shape,color_space)

            # resize the image
            res = pygame.transform.scale(res, (self.width, self.height))
            
            
            return True,res
        else:
            return False,res




    def led(self, option: str ="off"):
        # turn on or off the flash light
        if option =="on":
            return r.get(self.url+"/enabletorch")
        return r.get(self.url+"/disabletorch")

    def set_quality(self,option: int = 50):
        # Set the quality of the image
        # from 0 to 100
        if option > 100:
            option = 100
        if option < 0:
            option = 0
        return r.get(self.url +"/settings/quality?set={}".format(option))

    def set_orientation(self, orientation: int = 0):
        # Set the camera orientation
        # Landscape = 0
        # Portait = 1
        # Upside down = 2
        # Upside down portait = 3
        
        if orientation == 0:
            return r.get(f"{self.url}/settings/orientation?set=landscape")
        elif orientation == 1:
            return r.get(f"{self.url}/settings/orientation?set=portait")
        elif orientation == 2:
            return r.get(f"{self.url}/settings/orientation?set=upsidedown")
        elif orientation == 3:
            return r.get(f"{self.url}/settings/orientation?set=upsidedown_portait")



    def zoom(self, option: int = 0):
        if option < 0:
            option = 0
        if option > 100:
            option = 100
        return r.get(f"{self.url}/ptz?zoom={option}")
    
    def snapshot(self,img,blur_value):
        img_name = "img_{}.png".format(self.img_counter)
        path = os.path.join(PATH.BASE_PATH.value,PATH.SAVE_PATH_RAW_IMAGE.value)
        filename = os.path.join(path,img_name)
        
        
        cv2.imwrite(filename, img)
        cv2.putText(img, "CAPTURE", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255))
        # img_name = "img_capture.png"
        # cv2.imwrite(img_name,img)
        self.img_counter+=1
        # preprocess = PreprocessImage
        return img_name
        # return r.get((self.url + '/shot.jpg'))
    
    def connect_ipwebcam(self):
        if self.connect:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.connect = False
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_q:
                        self.connect = False
                        return AppState.HOME,None
                        # pygame.quit()
                    if event.key == pygame.K_l:
                        self.led("on")
                    if event.key == pygame.K_m:
                        self.led()
                    if event.key == pygame.K_g:
                        # Landscape
                        self.set_orientation(0)
                    if event.key == pygame.K_h:
                        # Portait
                        self.set_orientation(1)
                    if event.key == pygame.K_0:
                        is_img,img = self.get_image()
                        
                        #if not self.check_blurry(img):
                        img_name = self.snapshot(img,9)
                        # return AppState.RESULT,img_name
                        
                
                    if event.key == pygame.K_UP:
                        self.zoom_num += 25
                        if self.zoom_num > 100:
                            self.zoom_num = 100
                        self.zoom(self.zoom_num)
                    if event.key == pygame.K_DOWN:
                        self.zoom_num -= 25
                        if self.zoom_num < 0:
                            self.zoom_num = 0
                        self.zoom(self.zoom_num)
            # is_img,img = self.get_image()
          
            
            # return AppState.RESULT,img_name
            
            
        else:
            print('x')

    def draw(self,screen):
        is_success,img = self.get_pygame_image()

        if is_success:
            screen.blit(img,(0,0))
            # print('a')


        else:
            self.connect =False
            # return img
    
    def check_connection(self):
        return self.connect

    # threshold_min=12.5, threshold_max=13.6
    
    def check_blurry(self,img,threshold_min=18.1, threshold_max=19.5):

        blur_value = cv2.Laplacian(img, cv2.CV_64F).var()
        
        # print(base,blur_value)
        is_blur = blur_value > threshold_min and  blur_value < threshold_max

       


        # print()
        return is_blur,blur_value

    def check_brightness (self,img,threshold =128):
        is_enough_light = np.mean(img) > threshold
        return is_enough_light
    def get_error_msg(self):
        return self.msg
           




        
