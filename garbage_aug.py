import base64
import json
from labelme import utils
import cv2
import sys
import numpy as np
import random
import re
import os

if __name__ == "__main__":

    img_dir=os.listdir("./result/recyclable");
    for num in img_dir:
        img=cv2.imread("./result/recyclable/{}".format(num))
        
        dst1 = cv2.flip(img, 0)
        dst2 = cv2.flip(img, 1)
        dst3 = cv2.flip(img, -1)
        cv2.imwrite('./result/other_garbage/{}'.format(num +'flipx'+'.png'), dst1)
        cv2.imwrite('./result/other_garbage/{}'.format(num +'flipy'+'.png'), dst2)
        cv2.imwrite('./result/other_garbage/{}'.format(num +'flipxy'+'.png'), dst3)

        dst4 = cv2.GaussianBlur(img, (5, 5), 0)
        dst5 = cv2.GaussianBlur(dst1, (5, 5), 0)
        dst6 = cv2.GaussianBlur(dst2, (5, 5), 0)
        dst7 = cv2.GaussianBlur(dst3, (5, 5), 0)
        cv2.imwrite('./result/other_garbage/{}'.format(num +'_Gaussian'+'.png'), dst4)
        cv2.imwrite('./result/other_garbage/{}'.format(num +'flipx'+'Gaussian'+'.png'), dst5)
        cv2.imwrite('./result/other_garbage/{}'.format(num+'_flip_y'+'_Gaussian'+'.png'), dst6)
        cv2.imwrite('./result/other_garbage/{}'.format(num +'_flip_x_y'+'_Gaussian'+'.png'), dst7)
        height, width = img.shape[:2]
        factor1=random.uniform(0.5,3)
        factor2=random.uniform(0.5,3)
        factor3=random.uniform(0.5,3)
        factor4=random.uniform(0.5,3)
        factor5=random.uniform(0.5,3)
        factor6=random.uniform(0.5,3)

        size1 = (int(width*factor1), int(height*factor2))  
        size2 = (int(width*factor3), int(height*factor4))
        
        height, width = img.shape[:2]
        factor5=random.uniform(0.5,3)
        factor6=random.uniform(0.5,3)
        size3 = (int(width*factor5), int(height*factor6))
        img1 = cv2.resize(img, size1)
        img2 = cv2.resize(img, size2)
        img3 = cv2.resize(img, size3)
        cv2.imwrite('./result/other_garbage/{}'.format(num +'resize1'+'.png'), img1)
        cv2.imwrite('./result/other_garbage/{}'.format(num +'resize2'+'.png'), img2)
        cv2.imwrite('./result/recyclable/{}'.format(num +'resize3'+'.png'), img3)
