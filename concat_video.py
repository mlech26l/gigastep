import os

import cv2
import numpy as np

class FrameLoader:
    def __init__(self,path):
        self.frame_id = 0
        self.path = path

    def get(self):
        file = os.path.join(self.path,f"frame_{self.frame_id:04d}.png")
        if not os.path.exists(file):
            self.frame_id = 0
            file = os.path.join(self.path,f"frame_{self.frame_id:04d}.png")
        self.frame_id += 1
        return cv2.imread(file)


class FrameWriter:
    def __init__(self,path):
        self.frame_id = 0
        self.path = path
        os.makedirs(path,exist_ok=True)

    def write(self,*args):
        imgs = []
        for img in args:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img,(84*3,84*3),interpolation=cv2.INTER_NEAREST)
            imgs.append(img)
        imgs = np.concatenate(imgs,axis=1)
        file = os.path.join(self.path,f"frame_{self.frame_id:04d}.png")
        self.frame_id += 1
        cv2.imwrite(file,imgs)

l1 = FrameLoader("video/maps")
l2 = FrameLoader("video/visibility")
l3  = FrameLoader("video/many")
l4  = FrameLoader("video/hetero")

w = FrameWriter("video/concat")


for t in range(300):
    img1 = l1.get()
    img2 = l2.get()
    img3 = l3.get()
    img4 = l4.get()

    img1 = img1[:,:img1.shape[1]//2] # remove the right half
    img2 = img2[:,img2.shape[1]//2:] # remove the left half
    img3 = img3[:,:img3.shape[1]//2] # remove the right half
    img4 = img4[:,img4.shape[1]//2:] # remove the left half
    img4 = img4[4:-4,4:-4]
    img2 = img2[16:-16,16:-16]
    img3[:,img3.shape[1]-3:] = 255
    w.write(img1,img2,img3,img4)
    # convert -delay 4 -loop 0 -define webp:lossless=true video/concat/frame_*.png video/concat.webp