import cv2
import os
from keras.models import load_model
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.messagebox import showinfo
from tkinter import filedialog as fd


def fun1():
    global filename
    filename=select_file()
    #cap = cv2.imread(filename)
    img_arr = cv2.imread(filename)
    print("load model...")
    model = load_model('paddy_disease_detection.h5')
    #print( model.layers[0].input_shape)
    print("loaded")
    count=0
    x_val=[]
    
    fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)
    
    
    
    
    #ret, img_arr = cap.read()
    ax1.axis('off')
    ax1.imshow(img_arr)
    ax1.set_title('given image'+ str(img_arr.shape))
    
    
    try:
        img_arr=cv2.resize(img_arr,(224,224))
        img_arr = cv2.convertScaleAbs(img_arr, alpha=1.5, beta=0)
        img_arr = cv2.fastNlMeansDenoisingColored(img_arr,None,10,10,7,21)
        print("after preprocessing")
        cv2.imshow('PreProcess', img_arr)
        print(img_arr.shape)
            #print("after preprocessing")
            #plt.imshow(img_arr)
        x_val=[]
        x_val.append(img_arr)
        val_x=np.array(x_val)
        val_x=val_x/255.0
        predictions = model.predict(val_x, batch_size=32)
        print(predictions)
            
        results=predictions.argmax(axis=1)
        print(results)    
        accuracy=predictions.item(results[0]) *100
        #print("Accuracy : ", accuracy)
        classes=('BACTERIAL_LEAF_BLIGHT','BROWN_SPOT','HEALTHY','LEAF_BLAST','LEAF_SCALD','NARROW_BROWN_SPOT')
        rgb=((0,255,0),(255,0,0),(0,0,0),(0,0,255),(255,0,255),(0,255,255))
        print(classes[results[0]])
        frame=cv2.resize(img_arr,(512,512))
        height,width = frame.shape[:2] 
        thicc=2
        font = cv2.FONT_HERSHEY_TRIPLEX
        #frame=img_arr
        cv2.putText(frame, classes[results[0]],(1,height-50), font, 1,rgb[results[0]],1,cv2.LINE_AA)
        #cv2.rectangle(frame, (5,height//3), (width-5,height//2+20),rgb[results[0]],thicc)
        # load the model
        print(classes[results[0]])
    
        cv2.imshow("Paddy Leaf", frame)
       
        showinfo(title="result", message=classes[results[0]]) # + "\n" + "Accuracy of detection : " + str(accuracy) + "%")
                
        
    
    except :
        #print("error")
        pass
        
    
      



def select_file():
    filetypes = ( ('images files', '*.jpg'),       ('All files', '*.*')    )

    filename = fd.askopenfilename(   title='Open a file',     initialdir='./RiceLeafsDisease/samples/',        filetypes=filetypes)

    showinfo(         title='Selected File',        message=filename    )
    return filename



win=Tk()
win.geometry("550x350")
L1 = Label(win,text="PADDY LEAF DISEASE DETECTION",font=("bookman old style",20))
L1.grid(row=0,column=1,padx=10,pady=10)

b1 = Button(win,text="select image".center(42,' '),font=("bookman old style",20), bg="white", command=fun1)
b1.grid(row=1,column=1,padx=10,pady=10)


win.mainloop()
