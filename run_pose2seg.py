import numpy as np
from tqdm import tqdm

from pose2seg.modeling.build_model import Pose2Seg
from pose2seg.datasets.CocoDatasetInfo import CocoDatasetInfo, annToMask
from pycocotools import mask as maskUtils

import cv2
import os
import json

def get_masks(model, ImageRoot, AnnoFile, logger=print):    
    datainfos = CocoDatasetInfo(ImageRoot, AnnoFile, onlyperson=True, loadimg=True)
    
    model.eval()
    masks = {}
    for i in tqdm(range(len(datainfos))):
        rawdata = datainfos[i]
        img = rawdata['data']
        image_id = rawdata['id']
        
        height, width = img.shape[0:2]
		#groundtruth values from json file
        gt_kpts = np.float32(rawdata['gt_keypoints']).transpose(0, 2, 1) # (N, 17, 3)
        gt_segms = rawdata['segms']
            
        output = model([img], [gt_kpts])
        
        masks[image_id] = np.array(output[0])*255
    return masks

def get_seg_masks(images_path, keypoints_path, weights_path, output_path):
    print('===========> loading model <===========')
    model = Pose2Seg().cuda()
    model.init(weights_path)
            
    print('===========>   testing    <===========')

    #get masks
    masks = get_masks(model, images_path, keypoints_path)
    return masks
