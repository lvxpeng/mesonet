import io
from itertools import islice
from pathlib import Path

import deeplabcut
import pandas as pd
import yaml
from deeplabcut import stitch_tracklets
import pickle

from deeplabcut.utils import guarantee_multiindex_rows
from skimage.util import img_as_ubyte

# config_path = 'D:/deeplabcut/iosandgcamp-zju-2025-01-09/config.yaml'
# config_path = 'C:/Users/asus/Desktop/iosandgcamp-zju-2025-01-09/config.yaml'
config_path = 'C:/Users/asus/Desktop/iosandgcamp-zju-2025-01-09/config.yaml'
# config_path='C:/Users/asus/Desktop/iosandgcamp-zju-2025-01-09/training-datasets/iteration-0/UnaugmentedDataSet_iosandgcampJan9/metadata.yaml'
# videos='D:/temp/gcamp video/100-GaM.mp4'
videos='D:/deeplabcut/iosandgcamp-zju-2025-01-09/videos/未标注/mice_CON 6/S1 CON 6.avi'

# videos='D:/temp/gcamp video/DLC_atlas.avi'
# videos='D:/temp/gcamp video/O1_CON 2.avi'

# videotype='mp4'
videotype='avi'
save_as_csv=True
batchsize=1
allow_growth=True
auto_track=False
# pickle_file_path = "D:/temp/gcamp video/O1_CON 2DLC_Resnet50_iosandgcampJan9shuffle1_snapshot_220_full.pickle"


# deeplabcut.analyze_videos(config=config_path,videos=videos,shuffle=2,videotype=videotype,save_as_csv=True,batchsize=10)
# deeplabcut.create_video_with_all_detections(config=config_path,videos=videos,shuffle=2,videotype=videotype)
deeplabcut.create_labeled_video(config=config_path,videos=videos,shuffle=2,videotype=videotype)
# deeplabcut.convertcsv2h5(config_path,userfeedback=True,scorer='zju')
# with open(config_path,'r',encoding='utf-8') as stream:


#     config=yaml.safe_load(stream)
#     engine=config.get('train_fraction')
# print(engine)
# deeplabcut.refine_tracklets(config=config_path,pickle_or_h5_file='D:/temp/gcamp video/100-GaMDLC_Resnet50_iosandgcampJan9shuffle2_snapshot_220.h5',video=videos)
# try:
#     stitch_tracklets(config_path=config_path,videos=videos,videotype=videotype,split_tracklets=False,save_as_csv=True,track_method='ellipse')
# except Exception as e:
#     print(f"Error during stitching tracklets: {e}")

from mesonet import utils, natural_sort_key

# 测试查找keywords
# datapath=utils.findshuffle(config_path)
# print(datapath)
# index=utils.find_first_yaml_key(config_path,'TrainingFraction')
# index=utils.findfirstyaml_key(config_path,'multianimalproject')
#
# print(index)
# if index:
#     print('startrow={}'.format(3))
# else:
#     print('startrow={}'.format(2))

# 测试查找shuffle
# shuffle=utils.findshuffle(config_path)
# print('shuffle={}'.format(shuffle))

# with open(config_path, "r", encoding='utf-8') as config:
#     config = yaml.safe_load(config)
#     IsMulti = config.get("multianimalproject")
#     print('IsMulti={}'.format(IsMulti))

# import numpy as np
# import cv2
# import tifffile
# import glob
# import os
# import sys
# import skimage.io as  io
# from skimage.transform import resize
#
#
#
# img_array = []
#
# input_file='D:/temp/tif/tif/'
# output='D:/temp/tif/output/'
# size = (512, 512)
# coords_input_file=''
# # 读取TIF文件路径
# tif_list = glob.glob(os.path.join(input_file, "*.tif"))
#
# if tif_list:  # 是tif文件
#     '''
#     转换为png
#     '''
#     outputimagepath = os.path.join(output, 'tif_to_image')
#     tif_stack = tifffile.imread(tif_list[0])
#     number_images = len(tif_stack)
#     print('number_images:', number_images)
#
#     if not os.path.isdir(outputimagepath):
#         os.makedirs(outputimagepath)
#         print('output image dir:', outputimagepath)
#
#     for i in range(number_images):
#         image = tif_stack[i]
#         imageresized = resize(image, (512, 512))
#         imageunite8 = img_as_ubyte(imageresized)
#         outputfilename = os.path.join(outputimagepath, '{}.png'.format(i))
#         io.imsave(outputfilename, imageunite8)  # 保存到outputimagepath
#
#     # 所有TIFF图像都已转换并保存为PNG后，生成filenames列表
#     filenames = glob.glob(os.path.join(outputimagepath, "*.png"))
#     filenames.sort(key=natural_sort_key)
# else:
#     # 如果没有找到TIFF文件，则处理PNG文件
#     filenames = glob.glob(os.path.join(input_file, "*.png"))
#     filenames.sort(key=natural_sort_key)
#
# size = (512, 512)
# print(f"Number of PNG files found: {len(filenames)}")
# for filename in filenames:
#     print(filename)
#     img = cv2.imread(filename)
#     img = np.uint8(img)
#     img = cv2.resize(img, size)
#     img_array.append(img)
#
# if len(img_array) > 0:
#     video_output_path = os.path.join(output, "dlc_output")
#     video_name = os.path.join(video_output_path, "tmp_video.mp4")
#
#     if not os.path.isdir(video_output_path):
#         os.makedirs(video_output_path)
#
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(video_name, fourcc, 25, (512, 512))  #
#
#     try:
#         for i, image in enumerate(img_array):
#             if image.dtype != np.uint8:
#                 image = image.astype(np.uint8)
#             if image.shape[:2] != (512, 512):
#                 image = cv2.resize(image, (512, 512))
#             out.write(image)
#     except Exception as e:
#         print(f"Error writing video frame: {e}")
#     finally:
#         out.release()
#     print(f"Video saved to {video_name}")
# else:
#     print("No images found to create the video.")






#     for tif_file in tif_list:
#         img = tifffile.imread(os.path.join(input_file, tif_file))
#         img = np.uint8(img)
#         img = cv2.resize(img, size)
#         # 处理灰度转BGR
#         if len(img.shape) == 2:
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#         img_array.append(img)
#     # 更新size为实际尺寸
#     if img_array:
#         height, width, layers = img_array[0].shape
#         size = (width, height)
# else:
#     filenames = glob.glob(os.path.join(input_file, "*.png"))
#     filenames.sort(key=natural_sort_key)
#
#     size = (512, 512)
#     print(f"Number of PNG files found: {len(filenames)}")
#     for filename in filenames:
#         print(filename)
#         img = cv2.imread(filename)
#         img = np.uint8(img)
#         img = cv2.resize(img, size)
#         img_array.append(img)
#
# if len(img_array) > 0:
#     video_output_path = os.path.join(output, "dlc_output")
#     video_name = os.path.join(video_output_path, "tmp_video.mp4")
#
#     if not os.path.isdir(video_output_path):
#         os.makedirs(video_output_path)
#
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(video_name, fourcc, 1, (512, 512))
#
#     try:
#         for i, image in enumerate(img_array):
#             if image.dtype != np.uint8:
#                 image = image.astype(np.uint8)
#             if image.shape[:2] != (512, 512):
#                 image = cv2.resize(image, (512, 512))
#             out.write(image)
#     except Exception as e:
#         print(f"Error writing video frame: {e}")
#     finally:
#         out.release()
#     print(f"Video saved to {video_name}")
# else:
#     print("No images found to create the video.")