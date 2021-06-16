import json
import os
import cv2
import copy

def reorder_keypoints(keypoints):
        #keypoint correspondance list e.g. cList[5] = 10 means that the 5th keypoint in the coco format corresponds to the 10th in the raw format
        cList = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
    
        reordered_keypoints = []
        for i in range(len(cList)):
            #note that keypoints is a list in the following pattern [x,y,confidence, x, y, confidence, ...]
            #for coco the format is the same except the order of keypoints is different, and we use a status 0,1,2 instead of confidence (not used for us)
            for j in range(3):
                if j==2:
                    if keypoints[cList[i]*3+j] > 0:
                        reordered_keypoints.append(2)#status = 2 for inference
                    else:
                        reordered_keypoints.append(0)
                else:
                    reordered_keypoints.append(keypoints[cList[i]*3+j])
        return reordered_keypoints

def reformatter(raw_path, template_path, img_path, output_path):

    #load template
    with open(template_path) as f:
        templateDict = json.load(f)

    reformattedDict = {}
    reformattedDict["categories"] = templateDict["categories"]
    reformattedDict["images"] = []
    reformattedDict["annotations"] = []
    #find json file


    img_id = 0
    anno_id = 0

    for file in os.listdir(raw_path):
        if file.endswith(".json"):
            img_id += 1
            with open(os.path.join(raw_path, file)) as f:
                jsonDict = json.load(f)
            for person in jsonDict["people"]:
                #get template default key-values
                reformattedDict["images"] = reformattedDict["images"]+copy.deepcopy(templateDict["images"])
                reformattedDict["annotations"] = reformattedDict["annotations"]+copy.deepcopy(templateDict["annotations"])
                #insert file info
                ##file name
                img_ext = ""
                if file[:-15]+".jpg" in os.listdir(img_path):
                    img_ext = ".jpg"
                elif file[:-15]+".png" in os.listdir(img_path):
                    img_ext = ".png"
                assert img_ext == ".jpg" or img_ext == ".png", "Cannot find jpg or png image named"+file[1::-15]
                reformattedDict["images"][-1]["file_name"] = file[:-15] + img_ext
                ##image dimensions
                reformattedDict["images"][-1]["height"], reformattedDict["images"][-1]["width"], _ = cv2.imread(os.path.join(img_path,file[:-15]+img_ext)).shape
                ##image id
                reformattedDict["images"][-1]["id"] = img_id
                reformattedDict["annotations"][-1]["image_id"] = img_id
                ##re-ordered keypoints to match coco format
                reformattedDict["annotations"][-1]["keypoints"] = reorder_keypoints(person["pose_keypoints_2d"])
                ##annotation id
                anno_id += 1
                reformattedDict["annotations"][-1]["id"] = anno_id

    with open(os.path.join(output_path, "reformatted"+'.json'), 'w') as f:
        json.dump(reformattedDict, f)

if __name__ == "__main__":
    raw_path = os.path.join("data", "annotations", "raw")
    template_path = os.path.join("data", "annotations", "template", "template.json")
    img_path = os.path.join("data", "images")
    output_path = os.path.join("data", "annotations", "formatted")
    reformatter(raw_path, template_path, img_path, output_path)