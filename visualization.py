import argparse
import cv2
import json
import os
import numpy as np

def save_mask_over_image(masks, images_path, output_path):
    #apply mask on each image
    for file in os.listdir(images_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = cv2.imread(os.path.join(images_path,file))
            for mask in masks[file[:-4]]:
                #random mask color
                color = (np.random.random(size=3) * 256)
                colored_mask = np.zeros_like(img)
                for i in range(3):
                    colored_mask[:,:,i][mask>0.5] = color[i]
                img = cv2.addWeighted(img, 1, colored_mask, 0.5, 0)#hxwx3 vs hxwx1
            cv2.imwrite(os.path.join(output_path, file[:-4]+".png"), img)

def save_mask(masks, images_path, output_path, tag=None):

     for file in os.listdir(images_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            i = 0
            for mask in masks[file[:-4]]:
                i+=1
                cv2.imwrite(os.path.join(output_path, file[:-4] + "_" + tag + str(i) + ".png"), mask)

