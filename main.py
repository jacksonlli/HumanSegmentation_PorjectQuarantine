from file_transforms import json_reformatter, save_seg, get_input_paths, clear_folder, create_output_dir, copy_files
from run_openpose import write_json, write_images
from run_pose2seg import get_seg_masks
from run_gca_matting import get_mattes
from visualization import save_mask_over_image, save_mask
from image_transforms import get_trimaps, filter_masks, get_foreground_by_mask, merge_masks
import os

#change this to True if you want to output intermediary masks
def run(isKeepGroup=False, isSaveMasks = False):
    def inital_setup(images_path, raw_path, isSaveMasks=False):
        input_paths = get_input_paths()
        assert len(input_paths)>0
        clear_folder(images_path)
        copy_files(input_paths, images_path)
        clear_folder(raw_path)
        return create_output_dir(os.path.dirname(input_paths[0]), isSaveMasks)

    def alpha_matting_sequence(masks, images_path, present_mask_confidence, isSaveTrimap=False, trimap_name=None, sharpEdges=False):
        #--transform the mask into a trimap
        trimaps = get_trimaps(masks, present_mask_confidence = present_mask_confidence)
        ##--visualize trimaps
        if isSaveTrimap:
            save_mask(trimaps, images_path, masks_output_path, trimap_name)
        ##--get alpha mattes
        return get_mattes(images_path, trimaps=trimaps, sharpEdges=sharpEdges, img_max_area=2600000)

    images_path = os.path.join("data", "images")
    raw_path = os.path.join("data", "annotations", "raw")
    template_path = os.path.join("data", "annotations", "template", "template.json")
    formatted_path = os.path.join("data", "annotations", "formatted")
    weights_path = os.path.join("data", "weights", "pose2seg_release.pkl")
    keypoints_path =  os.path.join("data", "annotations", "formatted", "reformatted.json")
    output_path = "outputs"
    masks_output_path = os.path.join("outputs", "masks")


    ##get input images and setup directories
    output_path, masks_output_path = inital_setup(images_path, raw_path, isSaveMasks)
    ##--get pose from openpose
    write_json(images_path, raw_path)
    ##--reformat
    json_reformatter(raw_path, template_path, images_path, formatted_path)
    ##--get initial (rough) segmentation masks
    masks = get_seg_masks(images_path, keypoints_path, weights_path, output_path)
    ##--save masked images - optional
    if isSaveMasks:
        write_images(images_path, masks_output_path)
        save_mask_over_image(masks, images_path, masks_output_path)
        save_mask(masks, images_path, masks_output_path, "binary")
    ##--remove "small" masks
    masks = filter_masks(masks, 0.33, keypoints_path)
    ##if desired, merge the masks for each image
    if isKeepGroup:
        masks = merge_masks(masks)
    ##first pass to get a rough alpha matte
    mattes = alpha_matting_sequence(masks, images_path, 50, isSaveTrimap=isSaveMasks, trimap_name="trimap", sharpEdges=True)
    ##second pass with higher confidence/cleaner segmentation
    mattes = alpha_matting_sequence(mattes, images_path, 80, isSaveTrimap=isSaveMasks, trimap_name="trimap2_")
    ##--get final people segmentations
    segs = get_foreground_by_mask(mattes, images_path)
    ##--write pngs
    save_seg(segs, output_path, isKeepGroup)

if __name__ == "__main__":
    run(isKeepGroup = False)

