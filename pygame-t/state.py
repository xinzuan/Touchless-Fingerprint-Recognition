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
    FSRCNN_MODEL ="/home/vania/TA/Implement/Touchless-Fingerprint-Recognition/models/FSRCNN_x4.pb"
    EDSR_MODEL ="/home/vania/TA/Implement/Touchless-Fingerprint-Recognition/models/EDSR_x4.pb"
    ESGRAN_MODEL ="/home/vania/TA/Implement/Touchless-Fingerprint-Recognition/models/RRDB_ESRGAN_x4.pth"

    
    SAVE_PATH_RAW_IMAGE = '/home/vania/TA/Implement/Touchless-Fingerprint-Recognition/pygame-t/img'
    TEMP_PATH ='/home/vania/TA/Implement/Touchless-Fingerprint-Recognition/backend/complete/src/resources/temp/'
    SAVE_PATH ='/home/vania/TA/Implement/Touchless-Fingerprint-Recognition/pygame-t/res_preprocess'
    DB_PATH ='/home/vania/TA/Implement/Touchless-Fingerprint-Recognition/backend/complete/src/resources/fingerprints/'
        