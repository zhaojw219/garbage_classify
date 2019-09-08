
# coding: utf-8

import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np 
import pandas as pd 
from keras import optimizers

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from keras.applications import Xception
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

CATEGORIES = ['harmful_garbage', 'kitchen_garbage', 'other_garbage', 'recyclable']

# split validation set
split=False

if split:
    import random
    import shutil
    # make dir and mv
    os.mkdir('./dev/')
    for category in CATEGORIES:
        os.mkdir('./dev/' + category)
        name = os.listdir('/home/gnss/Desktop/result/' + category)
        random.shuffle(name)
        todev = name[:int(len(name) * .2)]
        for file in todev:
            shutil.move(os.path.join('/home/gnss/Desktop/result', category, file), os.path.join('dev', category))

# data generator 
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=[0.2,0.4],
    channel_shift_range=50,
    horizontal_flip=True,
    vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
    './result',
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical',
    shuffle=True)

val_datagen = ImageDataGenerator(rescale=1. / 255)

val_generator = val_datagen.flow_from_directory(
    './dev',
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical',
    shuffle=True)

tensorboard = TensorBoard('./logs')

# basic_model = Xception(include_top=False, weights='imagenet', pooling='avg')
basic_model = Xception(include_top=False, pooling='avg')

for layer in basic_model.layers:
    layer.trainable = True

input_tensor = basic_model.input
# build top
x = basic_model.output
x = Dropout(.5)(x)
print(len(CATEGORIES))
x = Dense(len(CATEGORIES), activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=x)

model_path='./logs/weights_xception.h5'
model.load_weights(model_path, by_name=True)
print('load weights')

adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='/home/gnss/Desktop/logs/weights_xception.h5',verbose=1,save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=3,verbose=1)

model.fit_generator(train_generator, steps_per_epoch=1000,epochs=1000, 
                    validation_steps=500,
                    validation_data=val_generator,
                    callbacks=[checkpointer,tensorboard,reduce_lr],
                    initial_epoch=40,
                    workers=4,
                    verbose=1)

model.save('xception.h5')



