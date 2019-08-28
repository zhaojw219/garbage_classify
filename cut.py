
import os
import cv2
import random

DATADIR="/home/gnss/Desktop/1.jpg"

'''
IMG_SIZE=300

IMG_SIZE = random.randint(1200,1500)

path=os.path.join(DATADIR) 

img_list=os.listdir(path)

print(img_list)

ind=0

for i in img_list:
    IMG_height = random.randint(1000,1300)
    IMG_width = random.randint(1000,1300)
    img_array=cv2.imread(os.path.join(path,i),cv2.IMREAD_COLOR)
    new_array=cv2.resize(img_array,(IMG_height,IMG_width))
    img_name=str(ind)+'.jpg'
    save_path='/home/gnss/Desktop/new_big_ship/'+str(ind)+'.jpg'
    ind=ind+1
    cv2.imwrite(save_path,new_array)
'''
w = 256
img = cv2.imread(DATADIR,-1)
img = cv2.resize(img,(w,w))
cv2.imwrite('/home/gnss/Desktop/2.jpg',img)
