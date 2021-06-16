import argparse
import cv2
import json
import os
import numpy as np

def visualize(masks, images_path, output_path, keypoints_path):
    #get image ids to match mask to image
    with open(os.path.join(keypoints_path)) as f:
        jsonDict = json.load(f)
    name2id = {}
    for image in jsonDict['images']:
        name2id[image['file_name']] = image['id']
    #apply mask on each image
    for file in os.listdir(images_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = cv2.imread(os.path.join(images_path,file))
            for mask in masks[name2id[file]]:
                #random mask color
                color = (np.random.random(size=3) * 256)
                colored_mask = np.zeros_like(img)
                for i in range(3):
                    colored_mask[:,:,i][mask>0.5] = color[i]
                img = cv2.addWeighted(img, 1, colored_mask, 0.5, 0)#hxwx3 vs hxwx1
            cv2.imwrite(os.path.join(output_path,file[:-4]+".png"), img)

if __name__ == "__main__":
    raw_path = os.path.join("data", "annotations", "raw")
    template_path = os.path.join("data", "annotations", "template", "template.json")
    images_path = os.path.join("data", "images")
    formatted_path = os.path.join("data", "annotations", "formatted")
    weights_path = os.path.join("data", "weights", "pose2seg_release.pkl")
    keypoints_path =  os.path.join("data", "annotations", "formatted", "reformatted.json")
    output_path = os.path.join("output", "masks")
    visualize(masks, images_path, output_path, keypoints_path)