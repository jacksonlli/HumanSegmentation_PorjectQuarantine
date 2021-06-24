import numpy as np
import os
import json
import cv2

def remove_background(img, mask, i):
    #https://stackoverflow.com/questions/61168140/opencv-removing-the-background-with-a-mask-image
    b, g, r = cv2.split(img)
    rgba = [b,g,r, mask]
    result = cv2.merge(rgba,4)
    return result

def write_pngs(images_path, keypoints_path, output_path, masks):
    #to get image id and name
    with open(os.path.join(keypoints_path)) as f:
        jsonDict = json.load(f)
    name2id = {}
    for image in jsonDict['images']:
        name2id[image['file_name']] = image['id']
    #apply mask on each image
    for file in os.listdir(images_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = cv2.imread(os.path.join(images_path,file))
            i = 0
            for mask in masks[name2id[file]]:
                i+=1
                new_img = remove_background(img, mask, i)
                cv2.imwrite(os.path.join(output_path,file[:-4]+"_"+str(i)+".png"), new_img)

def filter_masks(masks, scale_threshold, images_path, keypoints_path):
    #remove "small" masks (usually background people or floating heads)
    with open(os.path.join(keypoints_path)) as f:
        jsonDict = json.load(f)
    name2id = {}
    for image in jsonDict['images']:
        name2id[image['file_name']] = image['id']
    for file in os.listdir(images_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            #get tallest mask for an image and use it as size reference
            largest_h = 0
            h_list = []
            for mask in masks[name2id[file]]:
                contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                rect = cv2.boundingRect(contours[0])
                (x,y,w,h) = rect
                h_list.append(h)
                if h>largest_h:
                    largest_h = h
            keepBool = np.ones(len(masks[name2id[file]]), dtype=bool)
            for i in range(len(masks[name2id[file]])):
                h = h_list.pop(0)
                if h < largest_h*scale_threshold:
                    keepBool[i] = False
            masks[name2id[file]] = masks[name2id[file]][keepBool]
    return masks

def get_trimap(masks):
    k_size = 5
    present_mask_confidence = 30 # the larger it is, the smaller the grey(intermediate) zone is in the trimap and vice versa
    kernel = np.ones((k_size,k_size),np.uint8)
    for k, image_masks in masks.items():
        for mask in image_masks:
            contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rect = cv2.boundingRect(contours[0])
            (x,y,w,h) = rect
            num_iter = int(h/(k_size*present_mask_confidence))
            erosion = cv2.erode(mask, kernel, iterations = num_iter)
            dilation = cv2.dilate(mask, kernel, iterations = num_iter)
            mask[dilation>0] = 128
            mask[erosion>0] = 255
            cv2.imshow("mask", mask)
            break
    cv2.waitKey(0)

