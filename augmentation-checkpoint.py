# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:35:42 2024

@author: DARVIN MARY
"""
import numpy as np
import pandas as pd
from glob import glob
import os
import cv2

from PIL import *
count=0
train_path="E:/2024/B.E/PADDY/RiceLeafsDisease/train"
for folder in os.listdir(train_path):
    sub_path=train_path+"/"+folder
    for img in os.listdir(sub_path):
        image_path=sub_path+"/"+img
        #if(img!='augment'):
        #if(img!='augment2' and img!='augment'):
        if(img!='augment2' and img!='augment' and img!='augment3' ):
            img1=Image.open(image_path)
            #img1=img1.rotate(-45) 
            #img1=img1.rotate(45) 
            img1=img1.rotate(-45) 
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            #img1.show()
            #img1.save(train_path+"/"+folder+"/augment/"+img)
            #img1.save(train_path+"/"+folder+"/augment2/"+img)
            img1.save(train_path+"/"+folder+"/augment3/"+img)
            count+=1
            print(count)
print("File saved")