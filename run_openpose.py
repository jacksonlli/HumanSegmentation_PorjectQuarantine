import subprocess
import os
#only works for windows
#Double click on `openpose/models/getBaseModels.bat` to download the required body, face, and hand models.

def write_images():
    pipe = subprocess.call("cd openpose && bin\OpenPoseDemo.exe --image_dir ../data/images --write_images ../outputs/masks", shell=True)

def write_json(raw_path):
    pipe = subprocess.call("cd openpose && bin\OpenPoseDemo.exe --image_dir ../data/images --write_json ../"+raw_path+" --display 0 --render_pose 0", shell=True)

if __name__ == "__main__":
    raw_path = os.path.join("data", "annotations", "raw")
    write_json(raw_path)
    write_images()