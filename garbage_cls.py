import os
import numpy as np 
import pandas as pd 
from keras import optimizers
os.environ['CUDA_VISIBLE_DEVICES']='0'

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.applications import Xception
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

CATEGORIES = ['harmful_garbage', 'kitchen_garbage', 'other_garbage', 'recyclable']

split=True

if split:
    import random
    import shutil
    # make dir and mv
    os.mkdir('./dev/')
    for category in CATEGORIES:
        os.mkdir('./dev/' + category)
        name = os.listdir('./result/' + category)
        random.shuffle(name)
        todev = name[:int(len(name) * .2)]
        for file in todev:
            shutil.move(os.path.join('./result', category, file), os.path.join('dev', category))

'''
def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-4
    if epoch > 80:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
'''

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



basic_model = Xception(include_top=False, weights='imagenet', pooling='avg')
# basic_model = Xception(include_top=False, pooling='avg')

for layer in basic_model.layers:
    layer.trainable = True

input_tensor = basic_model.input
# build top
x = basic_model.output
x = Dropout(.5)(x)
print(len(CATEGORIES))
x = Dense(len(CATEGORIES), activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=x)

'''
model_path='./logs/weights_xception.h5'
model.load_weights(model_path, by_name=True)
print('load weights')
'''

adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


checkpointer = ModelCheckpoint(filepath='./logs/weights_xception.h5',verbose=1,save_best_only=True)

model.fit_generator(train_generator, steps_per_epoch=1000,epochs=1000, 
                    validation_steps=500,
                    validation_data=val_generator,
                    callbacks=[checkpointer,tensorboard],
                    workers=4,
                    verbose=1)

model.save('xception.h5')



