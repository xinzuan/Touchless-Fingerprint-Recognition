class ShowMessage:
    def __init__(self, text='', is_error=False):
        self.text = text
        self.is_error = is_error
    
    def set_msg(self,text):
        self.text = text