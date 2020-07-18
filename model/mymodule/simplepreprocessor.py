import numpy as np
import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image's  width, height and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter
        
    def preprocess(self, image):
        # resize the image to a fixed size (ignoring the aspect ratio)
        #ratio
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        r, th = cv2.threshold(gray,128, 255,cv2.THRESH_BINARY_INV)
        #im = np.expand_dims(th, axis = 2)
        return cv2.resize(th, (self.width, self.height), interpolation = self.inter)