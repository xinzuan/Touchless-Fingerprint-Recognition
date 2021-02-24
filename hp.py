
from urllib.request import urlopen

import cv2
# import matplotlib.pyplot as plt
# from skimage.feature import hessian_matrix, hessian_matrix_eigvals
# from skimage.io import imread, imshow
# import numpy as np 
# from skimage import filters
from threading import Thread


# host ="192.168.100.3"
# port="8080"
# username="13517090"
# password="13517090"
class VideoStreamWidget(object):
    def __init__(self, src=0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)
        self.img_counter =0

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def show_frame(self):
        # Display frames in main program
        if self.status:
            self.frame = self.maintain_aspect_ratio_resize(self.frame, width=600)
            roi=self.frame[100:300, 100:300]


            cv2.rectangle(self.frame,(100,100),(300,300),(255,255,1),0)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            cv2.putText(self.frame, "Press q to exit application", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
            cv2.putText(self.frame, "Press SPACE for capture image", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
            cv2.imshow('IP Camera Video Streaming', self.frame)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyWindow('IP Camera Video Streaming')
            # break
            #exit(1)
        elif key%256 == 32:
            # SPACE pressed
            img_name = "img_{}.jpg".format(self.img_counter)
            self.img_counter+=1
            cv2.imwrite(img_name,roi)
            # self.capture.release()
            # cv2.destroyWindow('IP Camera Video Streaming')
            return img_name

    # Resizes a image and maintains aspect ratio
    def maintain_aspect_ratio_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # Grab the image size and initialize dimensions
        dim = None
        (h, w) = image.shape[:2]

        # Return original image if no need to resize
        if width is None and height is None:
            return image

        # We are resizing height if width is none
        if width is None:
            # Calculate the ratio of the height and construct the dimensions
            r = height / float(h)
            dim = (int(w * r), height)
        # We are resizing width if height is none
        else:
            # Calculate the ratio of the 0idth and construct the dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # Return the resized image
        return cv2.resize(image, dim, interpolation=inter)
    def exitCamera(self):
        if self.capture.isOpened():
            self.capture.release()
            # cv2.destroyAllWindows()
            
       
# def imagecapturing(host ="192.168.100.3",port="8080"):



#     url = "http://"+host+":"+port+"/video"
# # url = "http://192.168.100.5:8090/video?username=13517090&password=13517090"

#     cam = cv2.VideoCapture(url)
#     img_counter = 0
#     while True:
#         ret, frame = cam.read()
#         #define roi which is a small square on screen
#         roi=frame[100:300, 100:300]


#         cv2.rectangle(frame,(100,100),(300,300),(255,255,1),0)
#         hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#         if not ret:
#             print("failed to grab frame")
#             break

#         # Processing of image and other stuff here
#         cv2.imshow('Frame', frame)
#         k = cv2.waitKey(1)
#         if k%256 == 27:
#             # ESC pressed
#             print("Escape hit, closing...")
#             break
#         elif k%256 == 32:
#             # SPACE pressed
#             img_name = "img_{}.jpg".format(img_counter)
#             cv2.imwrite(img_name, frame)
#             print("{} written!".format(img_name))

#             img_counter += 1

#     cam.release()

#     cv2.destroyAllWindows()
# while True:
#     print('b')
#     img_arr = np.array(bytearray(urlopen(url).read()),dtype=np.uint8)
#     print('a')
#     img = cv2.imdecode(img_arr,-1)
#     print('here')
#     cv2.imshow('IPWebcam',img)
    
#     if cv2.waitKey(1):
#         break
