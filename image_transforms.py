import numpy as np
import os
import json
import cv2

def remove_background(img, mask):
    #https://stackoverflow.com/questions/61168140/opencv-removing-the-background-with-a-mask-image
    b, g, r = cv2.split(img)
    rgba = [b,g,r, mask]
    result = cv2.merge(rgba,4)
    return result

def get_foreground_by_mask(masks, images_path):
    #apply mask on each image
    fg_imgs = {}
    for file in os.listdir(images_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = cv2.imread(os.path.join(images_path, file))
            fg_imgs[file[:-4]] = []
            for mask in masks[file[:-4]]:
                fg_imgs[file[:-4]].append(remove_background(img, mask))
    return fg_imgs            

def filter_masks(masks, scale_threshold, keypoints_path):
    #remove "small" masks (usually background people or floating heads)
    #first, get tallest mask for an image and use it as size reference
    for img_name, image_masks in masks.items():#masks is a dictionary where key is image id and value is all masks in image
        largest_h = 0
        h_list = []
        #get mask heights
        for mask in image_masks:
            contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rect = cv2.boundingRect(contours[0])
            (x,y,w,h) = rect
            h_list.append(h)
            if h>largest_h:
                largest_h = h
        keepBool = np.ones(len(image_masks), dtype=bool)
        #remove masks that are small
        for i in range(len(image_masks)):
            h = h_list.pop(0)
            if h < largest_h*scale_threshold:
                keepBool[i] = False
        masks[img_name] = image_masks[keepBool]
    return masks

def get_trimaps(masks, present_mask_confidence=30):
    # present_mask_confidence: the larger it is, the smaller the grey(intermediate) zone is in the trimap and vice versa
    k_size = 5
    kernel = np.ones((k_size,k_size),np.uint8)
    for k, image_masks in masks.items():
        for mask in image_masks:
            h = max(np.sum(mask, axis=0))
            num_iter = round(h/(k_size*present_mask_confidence*120))
            erosion = cv2.erode(mask, kernel, iterations = num_iter)
            dilation = cv2.dilate(mask, kernel, iterations = num_iter)
            mask = fill_edges(mask)
            mask[dilation>0] = 128
            mask[erosion>0] = 255
    cv2.waitKey(0)
    return masks

def fill_edges(mask, edge_width = 20):
    #pose2seg seems to always leave out the segmentation close to edges
    #this function fills any small gaps near edges

    #check which row/col of an edge needs filling
    top_edge = np.sum(mask[:edge_width, :], axis=0)>0
    top_layer = np.repeat(top_edge.reshape(1,-1), repeats=[edge_width], axis=0).reshape(edge_width,-1)
    mask[:edge_width, :][top_layer] = 128
    bottom_edge = np.sum(mask[-edge_width:, :], axis=0)>0
    bottom_layer = np.repeat(bottom_edge.reshape(1,-1), repeats=[edge_width], axis=0).reshape(edge_width,-1)
    mask[-edge_width:, :][bottom_layer] = 128
    left_edge = np.sum(mask[:, :edge_width], axis=1)>0
    mask[:, :edge_width][left_edge] = 128
    right_edge = np.sum(mask[:, -edge_width:], axis=1)>0
    mask[:, -edge_width:][right_edge] = 128

    return mask
