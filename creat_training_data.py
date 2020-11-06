import tensorflow as tf
import numpy as np
import rasterio
from read_and_write_RS_img import GRID

run=GRID()
proj_ms_big,geotrans_ms_big,data_ms,row1,column1 = run.read_img('ms.tif')
proj_ms,geotrans_ms,data_pan,row2,column2 = run.read_img('pan.tif')

ms = 'ms.tif'
pan = 'pan.tif'
ms_up = 'ms_up.tif'
pan_degraded = 'pan_d.tif'

with rasterio.open(ms) as f:
    metadata = f.profile
    ms_img = np.transpose(f.read(tuple(np.arange(metadata['count'])+1)),[1,2,0])
with rasterio.open(pan) as p:
    metadata = p.profile
    pan_img =np.transpose(p.read(tuple(np.arange(metadata['count'])+1)),[1,2,0])
with rasterio.open(ms_up) as q:
    metadata = q.profile
    ms_up_img =np.transpose(q.read(tuple(np.arange(metadata['count'])+1)),[1,2,0])
with rasterio.open(pan_degraded) as d:
    metadata = d.profile
    pan_degraded_img =np.transpose(d.read(tuple(np.arange(metadata['count'])+1)),[1,2,0])

train_data = np.concatenate((ms_up_img, pan_img), axis = 2)


train_data = np.transpose(train_data,[2,0,1])
train_ms = np.transpose(ms_img,[2,0,1])
train_pan = pan_img
train_pan_d = pan_degraded_img


sub_img_width=16
width=row2
height=column2
sub_img_num=width//sub_img_width*2-1
for i in range(sub_img_num):
    for j in range(sub_img_num):
        train_img = train_data[:,4*i*sub_img_width//2:4*(i+2)*sub_img_width//2,4*j*sub_img_width//2:4*(j+2)*sub_img_width//2]

        A = train_ms[:,i*sub_img_width//2:(i+2)*sub_img_width//2,j*sub_img_width//2:(j+2)*sub_img_width//2]
        B = train_pan[4*i*sub_img_width//2:4*(i+2)*sub_img_width//2,4*j*sub_img_width//2:4*(j+2)*sub_img_width//2]
        C = train_pan_d[i*sub_img_width//2:(i+2)*sub_img_width//2,j*sub_img_width//2:(j+2)*sub_img_width//2]
        label = A
        for q in range(4):
            for w in range(4):
                pan_cut_image = B[q*sub_img_width:(q+1)*sub_img_width,w*sub_img_width:(w+1)*sub_img_width]
                pan_cut_image = np.transpose(pan_cut_image,[2,0,1])
                label = np.concatenate((label, pan_cut_image), axis = 0)
        C = np.transpose(C,[2,0,1])
        label_img = np.concatenate((label,C),axis=0)


        run.write_img('training_data\\input\\img_{}.tif'.format(sub_img_num*i+j),proj_ms_big,geotrans_ms_big,train_img) 
        run.write_img('training_data\\label\\img_{}.tif'.format(sub_img_num*i+j),proj_ms,geotrans_ms_big,label_img) 



