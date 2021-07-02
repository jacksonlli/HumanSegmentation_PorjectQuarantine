import subprocess
import os
#only works for windows
#Double click on `openpose/models/getBaseModels.bat` to download the required body, face, and hand models.

def write_images(image_path, output_path):
    pipe = subprocess.call("cd openpose && bin\OpenPoseDemo.exe --image_dir ../"+image_path+" --write_images "+output_path, shell=True)

def write_json(image_path, raw_path):#raw_path is the path to the raw json
    pipe = subprocess.call("cd openpose && bin\OpenPoseDemo.exe --image_dir ../"+image_path+" --write_json ../"+raw_path+" --display 0 --render_pose 0", shell=True)

