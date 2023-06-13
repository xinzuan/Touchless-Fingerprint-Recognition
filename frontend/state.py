from enum import Enum

class AppState(Enum):
    QUIT = -1
    HOME = 0
    CAPTURE = 1
    UPLOAD = 2
    RESULT = 3


class Color(Enum):
    BLUE = (106, 159, 181)
    WHITE = (255, 255, 255)

class SRType(Enum):
    FSRCNN = 3,"fsrcnn"
    ESGRAN = 2
    EDSR = 1, "edsr"

class PATH(Enum):

    BASE_PATH = "/media/vania/Data/Touchless-Fingerprint-Recognition/"
    FSRCNN_MODEL ="../models/FSRCNN_x4.pb"
    EDSR_MODEL ="../models/EDSR_x4.pb"
    ESGRAN_MODEL ="../models/RRDB_ESRGAN_x4.pth"

   
    SAVE_PATH_RAW_IMAGE = 'frontend/img'
    TEMP_PATH ='backend/complete/src/resources/temp/'
    SAVE_PATH ='frontend/res_preprocess'
    DB_PATH ='backend/complete/src/resources/fingerprints/'


    SERVER_MATCH_URL ='http://localhost:8080/match'
    SERVER_ADD_URL ='http://localhost:8080/addUser'
        