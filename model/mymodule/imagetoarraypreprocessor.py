from tensorflow.keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    
    def __init__(self,  dataFormat = None):
        self.dataFormat = dataFormat
        
    def preprocess(self, image):
        return img_to_array(image, self.dataFormat)