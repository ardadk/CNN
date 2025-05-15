import os
import cv2
import urllib
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import random,os,glob
from imutils import paths
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from urllib.request import URLopener
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Conv2D,Flatten,MaxPooling2D,Dense,Dropout,SpatialDropout2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img,array_to_img
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        print(e)


dir_path = "C:/Users/Arda/Desktop/CNN/a"

target_size = (224,224)
waste_labels = {"cardboard":0,"glass":1,"metal":2,"paper":3,"plastic":4,"trash":5}
def load_datasets(path):
  x=[]
  labels=[]

  image_paths=sorted(list(paths.list_images(path)))

  for image_path in image_paths:
    img = cv2.imread(image_path)
    # Yeniden boyutlandırma için yüklenen görüntüyü kullanmak üzere 'image' yerine 'img' kullanıldı
    img = cv2.resize(img,target_size)
    x.append(img)
    label = image_path.split(os.path.sep)[-2]
    labels.append(waste_labels[label])
  x,labels = shuffle(x,labels,random_state=42)
  print(f"X boyutu: {np.array(x).shape}")
  print(f"Label sınıf sayısı: {len(np.unique(labels))} Gözlem sayısı: {len(labels)}")
  return x,labels
x,labels = load_datasets(dir_path)
input_shape =(np.array(x[0]).shape[1],np.array(x[0]).shape[1],3)
print(input_shape)
def visualize_img(image_batch,label_batch):
  plt.figure(figsize=(10,10))
  for n in range(10):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(np.array(list(waste_labels.keys()))[to_categorical(labels,num_classes=6)[n]==1][0].title())
      plt.axis('off')
visualize_img(x,labels)
train = ImageDataGenerator(horizontal_flip=True,
                           vertical_flip=True,
                           validation_split=0.1,
                           rescale=1./255,
                           shear_range=0.1,
                           zoom_range=0.1,
                           width_shift_range=0.1,
                           height_shift_range=0.1)
test = ImageDataGenerator(rescale=1./255,
                          validation_split=0.1)
train_generator = train.flow_from_directory(directory=dir_path,
                                           target_size=(target_size),
                                           batch_size=16,
                                           class_mode="categorical",
                                           subset="training")

test_generator = test.flow_from_directory(directory=dir_path,
                                           target_size=(target_size),
                                           batch_size=16,
                                           class_mode="categorical",
                                           subset="validation")
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),padding = "same",input_shape=input_shape,activation="relu"))
model.add(MaxPooling2D(pool_size=2,strides=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding = "same",input_shape=(input_shape),activation="relu"))
model.add(MaxPooling2D(pool_size=2,strides=(2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3),padding = "same",input_shape=(input_shape),activation="relu"))
model.add(MaxPooling2D(pool_size=2,strides=(2,2)))

model.add(Flatten())

model.add(Dense(units=64,activation="relu"))
model.add(Dropout(rate=0.2))
model.add(Dense(units=32,activation="relu"))
model.add(Dropout(rate=0.2))

model.add(Dense(units=6,activation="softmax"))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),"accuracy"])

callbacks= [EarlyStopping(monitor="val_loss",patience=50,verbose=1,mode="min"),
            ModelCheckpoint(filepath="mymodel.keras",monitor="val_loss",mode="min",save_best_only=True,save_weights_only=False,verbose=1)]

history = model.fit(train_generator,
                    epochs=100,
                    validation_data=test_generator,
                    callbacks=callbacks,
                    steps_per_epoch=2276//32,
                    validation_steps=251//32) 