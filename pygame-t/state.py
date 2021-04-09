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
    