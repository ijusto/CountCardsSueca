# adapted from https://www.instructables.com/id/Haar-Cascade-Python-OpenCV-Treinando-E-Detectando-/

import urllib.request
import numpy as np
import cv2
import os

with urllib.request.urlopen('http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03545756') as url:
    imgs_urls = url.read().decode()

img_num = 1

if not os.path.exists('imgs_dataset'):
    os.makedirs('imgs_dataset')

for url in imgs_urls.splitlines():
    try:
        print(url)
        urllib.request.urlretrieve(url, "imgs_dataset/"+str(img_num)+".jpg")
        img = cv2.imread("imgs_dataset/"+str(img_num)+".jpg",cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (100, 100))
        cv2.imwrite("imgs_dataset/"+str(img_num)+".jpg",img_resized)
        img_num += 1

    except Exception as e:
        print(str(e))