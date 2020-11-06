import glob
import gc
import numpy as np
import tensorflow as tf
from keras.models import Input,Model
from keras.layers import Conv2D
from keras.optimizers import Adam,SGD
import rasterio
from read_and_write_RS_img import GRID
# from tensor_TH import tensor_assign_2D
from keras.callbacks import ModelCheckpoint
import cv2
# from keras.models import load_model
from rebuild_function import image_model_predict
from function import loss_function

run=GRID()

DATA_DIR='training_data'
train_data = glob.glob(DATA_DIR+'\\input'+'\\img_*.tif')
label = glob.glob(DATA_DIR+'\\label'+'\\img_*.tif')

train_array_list = []
label_array_list = []


for file in range(len(train_data)):
    with rasterio.open(train_data[file]) as f:
        metadata = f.profile
        ms_img = np.transpose(f.read(tuple(np.arange(metadata['count'])+1)),[1,2,0])
        training_sample_array = ms_img
    train_array_list.append(training_sample_array)

    with rasterio.open(label[file]) as g:
        metadata = g.profile
        label_img = np.transpose(g.read(tuple(np.arange(metadata['count'])+1)),[1,2,0])
        label_sample_array = label_img
        #label_sample_array = label_img
    label_array_list.append(label_sample_array)


train_full_array = np.asarray(train_array_list)
label_full_array = np.asarray(label_array_list)


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
filepath = "checkpoint\\model_2.hdf5"

checkpoint = ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True)
model.fit(train_full_array,label_full_array,batch_size=32,epochs=1000,callbacks=[checkpoint])

model.save_weights('checkpoint\\model_2.hdf5')




