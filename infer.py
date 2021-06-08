import argparse
import numpy as np
from tqdm import tqdm

from modeling.build_model import Pose2Seg
from datasets.CocoDatasetInfo import CocoDatasetInfo, annToMask
from pycocotools import mask as maskUtils

import cv2
import os

def getMasks(model, ImageRoot, AnnoFile, logger=print):    
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
        
        masks[image_id] = output[0]
    return masks
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Pose2Seg inference")

    parser.add_argument(
        "--images",
        help="path to images folder",
        type=str,
    )

    args = parser.parse_args()
    
    print('===========> loading model <===========')
    model = Pose2Seg().cuda()
    weights_path = os.path.join("data", "weights", "pose2seg_release.pkl")
    model.init(weights_path)
            
    print('===========>   testing    <===========')

    keypoints_path =  os.path.join("data", "annotations", str(len(os.listdir(os.path.join("data", "annotations"))))+".json")

    #get masks
    masks = getMasks(model, args.images, keypoints_path)

    #apply mask on each image
    for file in os.listdir(args.images):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = cv2.imread(os.path.join(args.images,file))
            for mask in masks[int(file[:-4])]:
                #random mask color
                color = (np.random.random(size=3) * 256)
                colored_mask = np.zeros_like(img)
                for i in range(3):
                    colored_mask[:,:,i][np.array(mask)>0.5] = color[i]
                img = cv2.addWeighted(img, 1, colored_mask, 0.5, 0)#hxwx3 vs hxwx1
            cv2.imshow(file, img)
    cv2.waitKey(0)	