import argparse
import cv2
import json
import os
import numpy as np

#visualises json files formatted according the Pose2Seg specifications
#instructions: put json file and images in the same folder. Make sure the image_id in the json file corresponds to image name

if __name__=='__main__':
	parser = argparse.ArgumentParser(description="JSON visualization")
	parser.add_argument(
		"--path",
        help="Path to folder containing JSON file and images",
        type=str,
    )
	parser.add_argument(
		"--keypoints",
        help="Visualize keypoints",
        action="store_true",
    )

	parser.add_argument(
		"--mask",
        help="Visualize Segmentation Masks",
        action="store_true",
    )

	args = parser.parse_args()
	assert args.path, "Please enter path to folder containing [JSON file + image files]"
	assert args.mask or args.keypoints, "Please at select at least one element to visualize (--keypoints or --mask)"
	
	#find json file
	jsonFile = None
	for file in os.listdir(args.path):
		if file.endswith(".json"):
			jsonFile = file
			break
	assert jsonFile, "Please add JSON file to folder"

	#open json file and save it as a dict
	with open(os.path.join(args.path, jsonFile)) as f:
		jsonDict = json.load(f)
	 
	#create a dictionary that has the image id as keys and annotations as value
	annotations = {}
	for annotation in jsonDict["annotations"]:
		if annotation["image_id"] not in annotations:
			annotations[annotation["image_id"]] = []
		annotations[annotation["image_id"]].append(annotation)

	#display images

	for file in os.listdir(args.path):
		if file.endswith(".jpg") or file.endswith(".png"):
			img = cv2.imread(os.path.join(args.path,file))
			for annotation in annotations[int(file[:-4])]:
				color = (0,0,0)
				#pick a random "bright" color
				while (sum(list(color)) < 256):
					color = (np.random.random(size=3) * 256)
				if args.mask:
					#masks contain multiple masks, each mask is a list of repeating x and y pixel coordinates
					masks = annotation["segmentation"]
					img_mask = np.zeros_like(img)
					for mask in masks:
						i = 0
						pts = []
						while i < len(mask):
							pts.append([int(mask[i]), int(mask[i+1])])
							i+=2
						pts = np.array(pts).reshape((-1,1,2))
						cv2.fillPoly(img_mask,[pts],color)
					img = cv2.addWeighted(img, 1, img_mask, 0.5, 0)
				if args.keypoints:
					#keypoints is a list of repeating x, y, and status values where each triplet represents a point
					keypoints = annotation["keypoints"]
					i = 0
					while i < len(keypoints):
						#if status is not 0
						if keypoints[i+2] is not 0:
							cv2.circle(img, (keypoints[i], keypoints[i+1]), radius=0, color=color, thickness=5)
							cv2.circle(img, (keypoints[i], keypoints[i+1]), radius=4, color=(255,255,255), thickness=1)
						i += 3
					
			cv2.imshow(file, img)

	cv2.waitKey(0)	
