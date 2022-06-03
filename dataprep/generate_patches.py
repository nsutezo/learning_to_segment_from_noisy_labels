#Learn more or give us feedback
# This file generates patches for training
# -*- coding: UTF-8 -*-
"""
  @Email:  simonefobi@gmail.com

"""

import cv2
import numpy as np
import os
from functools import partial
import multiprocessing
import argparse
from skimage.draw import polygon
from skimage.io import imread, imsave
# import geopandas as gpd
from skimage.feature import register_translation 
from skimage.feature.register_translation import _upsampled_dft 
from scipy.ndimage import fourier_shift   


def patches_from_tif(args,imagename):
  image = cv2.imread(os.path.join(args.tifdir,args.cur_dir,'image',imagename))
  label = cv2.imread(os.path.join(args.tifdir,args.cur_dir,'label',imagename))
  print(" Starting with {}".format(imagename))
  for r in range(0,image.shape[0]-args.patchsize,args.patchsize):
    for c in range(0,image.shape[1]-args.patchsize,args.patchsize):
      img_patch = image[r:r+args.patchsize,c:c+args.patchsize,:]
      label_patch = label[r:r+args.patchsize,c:c+args.patchsize,:]
      # if (np.sum(np.all(img_patch!=[0,0,0], axis=-1)) / float(args.patchsize**2)) > args.threshold:
      if (np.sum(np.all(label_patch!=0, axis=-1)) / float(args.patchsize**2)) > args.threshold:
        savetoken =  imagename[:-4] + '_{}_{}_.png'.format(r,c)
        cv2.imwrite(os.path.join(args.patch_savedir,'images',savetoken),img_patch)
        cv2.imwrite(os.path.join(args.patch_savedir,'labels',savetoken),label_patch*225.0)
  print("Done with {}".format(imagename))

def compute_extent(lat, lon,center_x, center_y,row,col):
  res = 0.5 # m/px
  conversion = (0.0001 /11.132) * res # deg/px
  start_lon = lon - (center_x*conversion)
  stop_lon = lon + ((col-center_x)*conversion)

  bottom_lat =  lat - ((row - center_y)*conversion)
  top_lat = lat + (center_y*conversion)

  return start_lon, bottom_lat,stop_lon,top_lat


def convert_poly_to_rows_cols(poly, min_lon,max_lat):
  res = 1/0.5 # px/m
  conversion = (11.132/0.0001) *res  # px / deg
  lons, lats = poly.exterior.xy 
  lons =  [int((k - min_lon) * conversion) for k in lons]
  lats =  [int((max_lat - k ) * conversion) for k in lats]
  return lats, lons




def generate_from_airs(args):
  if not os.path.isdir(os.path.join(args.root_savedir,'airs')):
    os.makedirs(os.path.join(args.root_savedir,'airs'))
  
  for cur_dir in ['train' ,'val']:
    all_imagenames = os.listdir(os.path.join(args.tifdir,cur_dir,'image'))
    args.patch_savedir = os.path.join(args.root_savedir,'airs',cur_dir)
    args.cur_dir = cur_dir
    if not os.path.isdir(args.patch_savedir):
      os.makedirs(args.patch_savedir)
      os.makedirs(os.path.join(args.patch_savedir,'images'))
      os.makedirs(os.path.join(args.patch_savedir,'labels'))

    # import ipdb; ipdb.set_trace()
    # patches_from_tif(args,all_imagenames[0])
    # 
    pool = multiprocessing.Pool(processes=args.num_workers)
    func = partial(patches_from_tif,args)
    pool.map(func,all_imagenames)
    pool.close()
    pool.join()
    print('Done generating images for {}'.format(cur_dir))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description= 'Generating Data for training')
  
  args = parser.parse_args()
  args.patchsize = 128
  args.tifdir = '../../geoseg/dataset/trainval/'
  args.num_workers = 2
  args.threshold=0.01 #fraction of building area

  bgenerate_from_airs(args)