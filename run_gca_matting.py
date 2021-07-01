import os
import cv2
import toml
import argparse
import numpy as np
import os
import torch
from torch.nn import functional as F

import gca_matting.utils as utils
from   gca_matting.utils import CONFIG
import gca_matting.networks as networks

from gca_matting.demo import generator_tensor_dict, single_inference

#adapted from GCA-Matting/demo.py's __name__ = "__main__"
def get_mattes(image_dir, config = None, checkpoint = None, trimaps = None, sharpEdges=False, img_max_area = None):
    if config is None:
        config = os.path.join("gca_matting","config","gca-dist-all-data.toml")
    if checkpoint is None:
        checkpoint = os.path.join("gca_matting","checkpoints","gca-dist-all-data","gca-dist-all-data.pth")

    # Parse configuration
    with open(config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")

    # build model
    model = networks.get_generator(encoder=CONFIG.model.arch.encoder, decoder=CONFIG.model.arch.decoder)
    model.cuda()

    # load checkpoint
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

    # inference
    model = model.eval()
    for image_name in os.listdir(image_dir):
        # assume image and trimap have the same file name
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        i = 0
        for trimap in trimaps[image_name[:-4]]:
            #resize image if larger than what my gpu can handle
            w, h = trimap.shape
            if img_max_area and (w*h>img_max_area):
                w_new = int(w/np.sqrt(w*h/img_max_area))
                h_new = int(h/np.sqrt(w*h/img_max_area))
                trimap = cv2.resize(trimap, (h_new, w_new))
                image = cv2.resize(image, (h_new, w_new))
            image_dict = generator_tensor_dict(image=image, trimap=trimap)
            mattes, offset = single_inference(model, image_dict)
            if img_max_area and (w*h>img_max_area):
                trimaps[image_name[:-4]][i] = cv2.resize(mattes, (h,w))
            else:
                trimaps[image_name[:-4]][i] = mattes
            if sharpEdges:
                keepBool = trimaps[image_name[:-4]][i]>128
                removeBool = trimaps[image_name[:-4]][i]<=128
                trimaps[image_name[:-4]][i][keepBool]=255
                trimaps[image_name[:-4]][i][removeBool]=0
            i+=1
            torch.cuda.empty_cache()
    return trimaps#now alpha mattes
