# Import the libraries
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from mymodule.simpledatasetloader import SimpleDatasetLoader
from mymodule.imagetoarraypreprocessor import ImageToArrayPreprocessor
from mymodule.simplepreprocessor import SimplePreprocessor
from mymodule.lenet import LeNet
from mymodule.minivggnet import MiniVGGNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from imutils import paths
import numpy as np
import pandas as pd
import argparse
import cv2

# Define data path
data_path = 'data'

# Make data and label arrays
im_paths = list(paths.list_images(data_path))
sp = SimplePreprocessor(28,28)
sdl = SimpleDatasetLoader(preprocessors = [sp])
(data, labels) = sdl.load(im_paths, verbose = 500)

# Preprocess data and labels
data = np.expand_dims(data, axis = 3)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
data = data.astype('float32') / 255.0

# Split data and labels in training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size = 0.15)

# Define augmentation
data_gen_args = dict(width_shift_range=0.3,
                     height_shift_range=0.3,
                     zoom_range=0.5,
                     fill_mode='nearest')

#seed = 42

# Train data, provide the same seed and keyword arguments to the fit and flow methods
X_datagen = ImageDataGenerator(**data_gen_args)
X_datagen.fit(X_train)

# Val data, data augmentation, just for the sake of it
X_datagen_val = ImageDataGenerator(**data_gen_args)
X_datagen_val.fit(X_val)


#opt = SGD()

# Define model checkpoints to save the best models in terms of loss and accuracy
cp_1 = ModelCheckpoint('best_model_lenet_aug_loss_3.model', monitor = 'val_loss', mode = 'min',
                     save_best_only = True, verbose = 1)

cp_2 = ModelCheckpoint('best_model_lenet_aug_acc_3.model', monitor = 'val_acc', mode = 'max',
                     save_best_only = True, verbose = 1)

# Compile the model
model = LeNet.build(28, 28, 1, 9)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Start training
H = model.fit_generator(X_datagen.flow(X_train, y_train, batch_size=32),
                        validation_data = X_datagen_val.flow(X_val, y_val, batch_size=32),
                        callbacks = [cp_1, cp_2], steps_per_epoch = len(X_train) / 32,
                        epochs = 80)

