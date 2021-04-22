import pygame as pg
import pygame.freetype
from pygame.sprite import Sprite
from pygame.rect import Rect

from pygame.sprite import RenderUpdates
from state import Color

COLOR_INACTIVE = (106, 159, 181)
COLOR_ACTIVE = (255, 255, 255)


def create_surface_with_text(text, font_size, text_rgb, bg_rgb):
    """ Returns surface with text written on """
    font = pg.freetype.SysFont("Courier", font_size, bold=True)
    surface, _ = font.render(text=text, fgcolor=text_rgb, bgcolor=bg_rgb)
    return surface.convert_alpha()




class InputBox:

    def __init__(self, font,x, y, w, h,text=''):
        # print('e')
        self.rect = pg.Rect(x, y, w, h)
        self.color = COLOR_INACTIVE
        self.text = text
        self.font = font
        self.txt_surface = self.font.render(text, True, self.color)
        self.active = False
        

    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            # If the user clicked on the input_box rect.
            if self.rect.collidepoint(event.pos):
                # Toggle the active variable.
                self.active = not self.active
            else:
                self.active = False
            # Change the current color of the input box.
            self.color = COLOR_ACTIVE if self.active else COLOR_INACTIVE
        if event.type == pg.KEYDOWN:
            if self.active:
                if event.key == pg.K_RETURN:
                    print(self.text)
                    self.text = ''
                elif event.key == pg.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text.
                self.txt_surface = self.font.render(self.text, True, self.color)

    def update(self):
        # Resize the box if the text is too long.
        width = max(200, self.txt_surface.get_width()+10)
        self.rect.w = width

    def draw(self, screen):
        # Blit the text.
        
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        # Blit the rect.
        pg.draw.rect(screen, self.color, self.rect, 5)
       
    def get_text(self):
        return self.text

class UIElement(Sprite):
    """ An user interface element that can be added to a surface """

    def __init__(self, center_position, text, font_size, bg_rgb, text_rgb, action=None,hover=True):
        """
        Args:
            center_position - tuple (x, y)
            text - string of text to write
            font_size - int
            bg_rgb (background colour) - tuple (r, g, b)
            text_rgb (text colour) - tuple (r, g, b)
            action - the gamestate change associated with this button
        """
        self.mouse_over = False
        

        default_image = create_surface_with_text(
            text=text, font_size=font_size, text_rgb=text_rgb, bg_rgb=bg_rgb
        )
        
        highlighted_image = create_surface_with_text(
            text=text, font_size=font_size * 1.2, text_rgb=text_rgb, bg_rgb=bg_rgb
        )

        self.images = [default_image, highlighted_image]

        self.rects = [
            default_image.get_rect(center=center_position),
            highlighted_image.get_rect(center=center_position),
        ]

        self.action = action
        self.hover = hover

        super().__init__()

    @property
    def image(self):
        return self.images[1] if self.mouse_over and self.hover else self.images[0]

    @property
    def rect(self):
        return self.rects[1] if self.mouse_over and self.hover else self.rects[0]

    def update(self, mouse_pos, mouse_up):
        """ Updates the mouse_over variable and returns the button's
            action value when clicked.
        """
        if self.rect.collidepoint(mouse_pos):
            self.mouse_over = True
            if mouse_up:
                return self.action
        else:
            self.mouse_over = False

    def draw(self, surface):
        """ Draws element onto a surface """
        surface.blit(self.image, self.rect)



    
        

class ImageResult:
    def __init__(self, filename,width =100,height = 100):
        # print('e')
        self.file = pg.image.load(filename)
        self.width = width
        self.height =height

        # resize :
        # https://stackoverflow.com/questions/20002242/how-to-scale-images-to-screen-size-in-pygame
        



    def draw(self, screen):
       
        
        screen.blit(pg.transform.scale(self.file, (self.width, self.height)), (50, 50))



