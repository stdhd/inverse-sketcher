import cv2
import os
from tqdm import tqdm

"""Split the images of the edges2shoes dataset into sketch and photo"""


dirs = ["train", "val"]

if not os.path.exists("sketch"):
    os.mkdir("sketch")
if not os.path.exists("photo"):
    os.mkdir("photo")

for dir in dirs:
    print("Converting {} folder".format(dir))
    imgs = os.listdir(dir)
    for img in tqdm(imgs):
        im = cv2.imread(os.path.join(dir, img))
        sketch, photo = cv2.flip(im[:,:im.shape[1]//2], 1), cv2.flip(im[:,im.shape[1]//2:], 1)
        cv2.imwrite(os.path.join("sketch", img), sketch)
        cv2.imwrite(os.path.join("photo", img), photo)
