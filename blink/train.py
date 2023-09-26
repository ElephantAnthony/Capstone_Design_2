import pandas as pd
import datetime

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau



plt.style.use('dark_background')

# CSV 파일 불러오기
data = pd.read_csv('../eye_blink_detector-master/dataset/dataset.csv')

# 데이터프레임 출력
print(data.head())


# load dataset
x_train = np.load('../eye_blink_detector-master/dataset/x_train.npy').astype(np.float32)
y_train = np.load('../eye_blink_detector-master/dataset/y_train.npy').astype(np.float32)
x_val = np.load('../eye_blink_detector-master/dataset/x_val.npy').astype(np.float32)
y_val = np.load('../eye_blink_detector-master/dataset/y_val.npy').astype(np.float32)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)


# data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_genrator = train_datagen.flow(
    x = x_train, y = y_train,
    batch_size = 32,
    shuffle=True
)

val_generator = val_datagen.flow(
    x = x_val, y = y_val,
    batch_size = 32,
    shuffle = False
)

# build model
inputs = Input(shape=(26, 34, 1))

net = Conv2D(32, kernel_size = 3, strides=1, padding='same', activation='relu')(inputs)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(64, kernel_size = 3, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(128, kernel_size = 3, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Flatten()(net)

net = Dense(512)(net)
net = Activation('relu')(net)
net = Dense(1)(net)
outputs = Activation('sigmoid')(net)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()


# train
start_time = datetime.datetime.now().strftime('%Y_%m_%H_%M_%S')

model.fit(
    train_genrator, epochs=50, validation_data = val_generator,
    callbacks=[
        ModelCheckpoint('models/%s.h5' %(start_time), monitor='val_acc', save_best_only=True, mode='max',
                       verbose=1),
        ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)
    ]
)

