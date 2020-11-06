import cv2
import gc
import glob
import numpy as np
import rasterio

def image_model_predict(input_ms_image_filename, input_pan_image_filename, img_height_size, img_width_size, fitted_model,
                        write, output_filename):
    """
    This function cuts up an image into segments of fixed size, and feeds each segment to the model for upsampling. The
    output upsampled segment is then allocated to its corresponding location in the image in order to obtain the complete upsampled
    image, after which it can be written to file.

    Inputs:
    - input_ms_image_filename: File path of the multispectral image to be pansharpened by the PCNN model
    - input_pan_image_filename: File path of the panchromatic image to be used by the PCNN model
    - img_height_size: Height of image segment to be used for PCNN model pansharpening
    - img_width_size: Width of image segment to be used for PCNN model pansharpening
    - ms_to_pan_ratio: The ratio of pixel resolution of multispectral image to that of panchromatic image
    - fitted_model: Keras model containing the trained PCNN model along with its trained weights
    - write: Boolean indicating whether to write the pansharpened image to file
    - output_filename: File path to write the file

    Output:
    - pred_img_actual: Numpy array which represents the pansharpened image

    """

    with rasterio.open(input_ms_image_filename) as f:
        metadata = f.profile
        ms_img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])

    with rasterio.open(input_pan_image_filename) as g:
        metadata_pan = g.profile
        pan_img = g.read(1)

    pan_img = np.expand_dims(pan_img, axis = 2)

    #ms_to_pan_ratio = metadata['transform'][0] / metadata_pan['transform'][0]
    ms_to_pan_ratio = 1

    ms_img_upsampled = cv2.resize(ms_img, (int(ms_img.shape[1] * ms_to_pan_ratio), int(ms_img.shape[0] * ms_to_pan_ratio)),
                                  interpolation = cv2.INTER_CUBIC)

    pred_stack = np.concatenate((ms_img_upsampled, pan_img), axis = 2)

    y_size = ((pred_stack.shape[0] // img_height_size) + 1) * img_height_size
    x_size = ((pred_stack.shape[1] // img_width_size) + 1) * img_width_size

    if (pred_stack.shape[0] % img_height_size != 0) and (pred_stack.shape[1] % img_width_size == 0):
        img_complete = np.zeros((y_size, pred_stack.shape[1], pred_stack.shape[2]))
        img_complete[0 : pred_stack.shape[0], 0 : pred_stack.shape[1], 0 : pred_stack.shape[2]] = pred_stack
    elif (pred_stack.shape[0] % img_height_size == 0) and (pred_stack.shape[1] % img_width_size != 0):
        img_complete = np.zeros((pred_stack.shape[0], x_size, pred_stack.shape[2]))
        img_complete[0 : pred_stack.shape[0], 0 : pred_stack.shape[1], 0 : pred_stack.shape[2]] = pred_stack
    elif (pred_stack.shape[0] % img_height_size != 0) and (pred_stack.shape[1] % img_width_size != 0):
        img_complete = np.zeros((y_size, x_size, pred_stack.shape[2]))
        img_complete[0 : pred_stack.shape[0], 0 : pred_stack.shape[1], 0 : pred_stack.shape[2]] = pred_stack
    else:
         img_complete = pred_stack

    pred_img = np.zeros((img_complete.shape[0], img_complete.shape[1], ms_img.shape[2]))

    img_holder = np.zeros((1, img_height_size, img_width_size, img_complete.shape[2]))
    preds_list = []

    for i in range(0, img_complete.shape[0], img_height_size):
        for j in range(0, img_complete.shape[1], img_width_size):
            img_holder[0] = img_complete[i : i + img_height_size, j : j + img_width_size, 0 : img_complete.shape[2]]
            preds = fitted_model.predict(img_holder)
            preds_list.append(preds)

    n = 0
    for i in range(0, pred_img.shape[0], img_height_size):
            for j in range(0, pred_img.shape[1], img_width_size):
                pred_img[i : i + img_height_size, j : j + img_width_size, 0 : ms_img.shape[2]] = preds_list[n]
                n += 1

    pred_img_actual = np.transpose(pred_img[0 : pred_stack.shape[0], 0 : pred_stack.shape[1], 0 : ms_img.shape[2]],
                                   [2, 0, 1]).astype(metadata_pan['dtype'])

    metadata_pan['count'] = ms_img_upsampled.shape[2]
    with rasterio.open(output_filename, 'w', **metadata_pan) as dst:
        dst.write(pred_img_actual)

    return pred_img_actual
