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

def get_seg(images_path, keypoints_path, weights_path, output_path):
    #filter threshold: discard masks smaller than a % of the largest mask. 
    #e.g. 0.33 will remove any mask shorter than a third of the height of the tallest mask
    print('===========> loading model <===========')
    model = Pose2Seg().cuda()
    model.init(weights_path)
            
    print('===========>   testing    <===========')

    #get masks
    masks = get_masks(model, images_path, keypoints_path)
    return masks


                    
                

if __name__ == "__main__":
    weights_path = os.path.join("data", "weights", "pose2seg_release.pkl")
    keypoints_path =  os.path.join("data", "annotations", "formatted", "reformatted.json")
    images_path =  os.path.join("data", "images")

    get_seg(images_path, keypoints_path, weights_path)