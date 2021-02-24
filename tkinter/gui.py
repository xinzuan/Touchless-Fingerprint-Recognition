from PIL import ImageTk, Image as PILImage
import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog, messagebox
from tkinter.messagebox import *
from tkinter.filedialog import asksaveasfile
import importlib, importlib.util
import wave
import pyaudio
import threading
import cv2


import numpy as np
from cv2 import cv2

LARGEFONT = ("Verdana", 35)

MEDIA_CHOICE = {
    "ImageLSB"
}

# constant

START_ROW = 2
START_COLUMN = 0
KEY_ROW = 6
STEGANO_ROW = 8


# import utils
def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# import

img= module_from_file("hp", "../hp.py")
utils = module_from_file("preprocess","../preprocess.py")


class tkinterApp(tk.Tk):
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):

        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)

        # creating a container
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # initializing frames to an empty array
        self.frames = {}

        # iterating through a tuple consisting
        # of the different page layouts
        for F in (ImageLSBPage,ImageLSBPage):

            frame = F(container, self)

            # initializing frame of that object from
            # ImagePage, VideoPage, AudioPage respectively with
            # for loop
            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(ImageLSBPage)

    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    # quit
    def quit(self):
        self.root.destroy()


class BasePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)


        self.winfo_toplevel().title("Recognition")




class ImageLSBPage(BasePage):
    result = None

    def __init__(self, parent, controller):
        super().__init__(parent, controller)

        label = ttk.Label(self, text="Recognition", font=("Verdana", 15))
        label.grid(row=1, column=1, padx=10, pady=(20, 20))

        # Stego Media
        self.stego_media_path = None

        take_picture_button = tk.Button(
            self, text="Take Picture", command=self.image_capturing
        ).grid(row=START_ROW+1, column=START_COLUMN+4, sticky=tk.W, pady=4)







        # Insert Host
        tk.Label(self, text="Host :").grid(row=START_ROW, column=START_COLUMN, sticky=W, padx=(20, 0))
        self.host_input = tk.Entry(self)
        self.host_input.insert(tk.END, "192.168.100.6")
        self.host_input.grid(row=START_ROW + 1, column=START_COLUMN, sticky=W, padx=(20, 0))

        # Insert Port
        tk.Label(self, text="Port:").grid(row=START_ROW, column=START_COLUMN+1, sticky=W, padx=(20, 0))
        self.port_input = tk.Entry(self)
        self.port_input.grid(row=START_ROW+ 1, column=START_COLUMN+1, sticky=W, padx=(20, 0))

        # Start Button
        start_btn = tk.Button(
            self, text="Start comparing", command=self.extract_stego_obj
        ).grid(row=7, column=1, sticky=tk.W, pady=4)

        # Exit  Button
        exit_btn = tk.Button(
            self, text="Close", command=quit
        ).grid(row=7, column=2, sticky=tk.W, pady=4)


        # Message for User
        tk.Label(self, text="Message:").grid(row=10, column=0, sticky=E, pady=20)
        self.user_message = tk.Label(self, text="No message right now")
        self.user_message.grid(row=10, column=1, sticky=W, pady=20)

    def image_capturing(self):
      
        
        if ( self.get_host() and self.get_port()):
            self.host = self.get_host()
            self.port = self.get_port()
            stream_link = "http://"+self.host+":"+self.port+"/video"
            try:
                video_stream_widget = img.VideoStreamWidget(stream_link)
            except:
                self.set_message("Error mengkoneksi ipwebcam dengan aplikasi")
            while True:
                try:
                    image_taken = video_stream_widget.show_frame()
                    if(image_taken):
                        print("halo")
                       
                       # video_stream_widget.exitCamera()
                       
                        preprocess_utils = utils.PreprocessImage(image_taken)
                        max_ridges, min_ridges = preprocess_utils.detect_ridges()
                        binary_res_name = preprocess_utils.threshold()
                        binary_res = cv2.imread(binary_res_name)
                        preprocess_utils.plot_images(max_ridges,binary_res)
                        print("here")
                except AttributeError:
                    pass
        else:
            self.set_message("Masukkan host dan port terlebih dahulu")
        
        
        # self.img = img.imagecapturing(self.get_host(),self.get_port())
        # pil_image = self.lsb.read_stego_media(self.stego_media_path)

        # self.set_original_image(pil_image)

    def upload_message_file(self):
        message_path = filedialog.askopenfilename(filetypes=[("Any File", "*")])
        if not message_path:
            return

        try:
            self.lsb.insert_message(message_path)
            self.message_path = message_path
            self.message_label["text"] = " : " + self.message_path
        except Exception as error:
            self.set_message(str(error))

    def produce_stego_obj(self):
        user_key = self.get_user_key()
        enable_encrypt = not not self.enable_encryption.get()
        randomize = not not self.randomize.get()
        if (enable_encrypt or randomize) and (not user_key):
            self.set_message("Please provide key when enabling encryption or randomizing")
            return

        result = self.lsb.produce_stego_obj(enable_encrypt, randomize, user_key)
        self.set_result_image(result)
        self.result = result

    def extract_stego_obj(self):
        stego_obj_path = filedialog.askopenfilename(filetypes=[("Image Files", ".png .bmp")])
        if not stego_obj_path:
            return
        # self.stego_media_path = media_path
        # self.stego_media_label["text"] = " : " + self.stego_media_path

        user_key = self.get_user_key()

        binary_content, file_name = lsb.LSB().read_stego_image(stego_obj_path, user_key)
        self.set_message("{} has been extracted".format(file_name))

        save_path = filedialog.asksaveasfilename(initialfile=file_name)
        if save_path:
            binary_file = open(save_path, "wb")
            binary_file.write(binary_content)
            binary_file.close()

    def set_original_image(self, pil_image):
        # Original Photo
        print("photo")
        tk.Label(self, text="Original Image").grid(row=11, column=0)
        pil_image = pil_image.resize((200, 200), PILImage.ANTIALIAS)
        display_img = ImageTk.PhotoImage(image=pil_image)
        self.original_photo = Label(self, image=display_img)
        self.original_photo.img = display_img
        self.original_photo.grid(row=12, column=0)

    def set_result_image(self, pil_image):
        # Stego Photo
        tk.Label(self, text="Stego Result").grid(row=11, column=1)
        pil_image = pil_image.resize((200, 200), PILImage.ANTIALIAS)
        display_img = ImageTk.PhotoImage(image=pil_image)
        self.result_photo = Label(self, image=display_img)
        self.result_photo.img = display_img
        self.result_photo.grid(row=12, column=1, sticky=W + E + N + S)

    def save_result_as(self):
        if not self.result:
            self.set_message("No result exist currently")
            return
        message_path = filedialog.asksaveasfilename()
        if message_path:
            self.result.save(message_path)

    def get_host(self):
        return self.host_input.get()

    def set_message(self, message):
        self.user_message["text"] = message

    def get_port(self):
        return self.port_input.get()



# Driver Code
app = tkinterApp()
app.mainloop()

# paaaaSword1233