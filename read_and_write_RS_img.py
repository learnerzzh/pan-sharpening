from osgeo import gdal
import os
import numpy as np
import rasterio

class GRID:

    def read_img(self,filename):
        dataset=gdal.Open(filename)
        im_width = dataset.RasterXSize
        im_height = dataset.RasterYSize
        im_geotrans = dataset.GetGeoTransform()
        im_proj = dataset.GetProjection()
        im_data = dataset.ReadAsArray(0,0,im_width,im_height)
        del dataset
        return im_proj,im_geotrans,im_data,im_width,im_height


    def write_img(self,filename,im_proj,im_geotrans,im_data):

        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_UInt16

        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1,im_data.shape

        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  #
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])
        del dataset
