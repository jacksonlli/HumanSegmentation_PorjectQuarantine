from json_reformatter import reformatter
from run_openpose import write_json
from run_pose2seg import get_seg
from visualization import mask_over_image, mask_binary
from utils import write_pngs, get_trimap, filter_masks
import os


raw_path = os.path.join("data", "annotations", "raw")
template_path = os.path.join("data", "annotations", "template", "template.json")
images_path = os.path.join("data", "images")
formatted_path = os.path.join("data", "annotations", "formatted")
weights_path = os.path.join("data", "weights", "pose2seg_release.pkl")
keypoints_path =  os.path.join("data", "annotations", "formatted", "reformatted.json")
output_path = "outputs"
masks_output_path = os.path.join("outputs", "masks")

#clear past files
for file in os.listdir(raw_path):
    os.remove(os.path.join(raw_path, file))

##get pose from openpose
write_json(raw_path)
##reformat
reformatter(raw_path, template_path, images_path, formatted_path)
##get segmentation masks
masks = get_seg(images_path, keypoints_path, weights_path, output_path)
##save masked images - optional
#mask_over_image(masks, images_path, masks_output_path, keypoints_path)
#mask_binary(masks, images_path, masks_output_path, keypoints_path)
##remove "small" masks
masks = filter_masks(masks, 0.33, images_path, keypoints_path)
##transform the mask into a trimap
trimaps = get_trimap(masks)
##get png images
#write_pngs(images_path, keypoints_path, output_path, trimaps)

