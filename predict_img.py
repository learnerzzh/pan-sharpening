from keras.models import Input,Model,load_model
from keras.layers import Conv2D,Flatten,Dense,Reshape,Conv2DTranspose
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from read_and_write_RS_img import GRID
import glob
import rasterio
import tensorflow as tf
from rebuild_function import image_model_predict
from function import loss_function
run=GRID()

def improved_SRCNN_model(image_height_size,image_width__size,n_bands,n1=64,n2=32,n3=16,f1=9,f2=7,f3=5,f4=3,l_r=0.0001):
    img_input = Input(shape = (image_height_size,image_width__size,n_bands))
    conv1 = Conv2D(n1,(f1,f1),padding = 'same',activation='relu')(img_input)
    conv2 = Conv2D(n2,(f2,f2),padding = 'same',activation='relu')(conv1)
    conv3 = Conv2D(n3,(f3,f3),padding = 'same',activation='relu')(conv2)
    conv4 = Conv2D(n_bands-1,(f4,f4),padding = 'same')(conv3)


    model = Model(inputs=img_input,outputs=conv4)
    model.compile(optimizer = Adam(lr=l_r),loss = loss_function ,metrics=[loss_function])
    return model

model = improved_SRCNN_model(64,64,9)

model.load_weights('checkpoint\\model_2.hdf5')

input_ms_image_filename = 'image\\ms_up.tif'
input_pan_image_filename = 'image\\pan.tif'
train_size = 64
output_filename = 'image\\fused_t1.tif'

image_model_predict(input_ms_image_filename, input_pan_image_filename, train_size, train_size, fitted_model=model,write=True, output_filename=output_filename)


