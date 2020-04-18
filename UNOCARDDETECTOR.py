import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
# baseline cnn model for mnist
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import numpy 
import pandas 
import glob
import matplotlib.pylab as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LSTM
from keras.layers import Activation, Flatten, Input, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D

def takePictureFromVideo():
    video = cv2.VideoCapture(0)
    while True:
        check,frame = video.read()
        frame2=frame
        cv2.line(img=frame2, pt1=(0, 240), pt2=(720,240), color=(255, 0, 0), thickness=5, lineType=8, shift=0)
        cv2.line(img=frame2, pt1=(90, 240), pt2=(90,480), color=(255, 0, 0), thickness=5, lineType=8, shift=0)
        cv2.line(img=frame2, pt1=(180, 240), pt2=(180,480), color=(255, 0, 0), thickness=5, lineType=8, shift=0)
        cv2.line(img=frame2, pt1=(270, 240), pt2=(270,480), color=(255, 0, 0), thickness=5, lineType=8, shift=0)
        cv2.line(img=frame2, pt1=(360, 240), pt2=(360,480), color=(255, 0, 0), thickness=5, lineType=8, shift=0)
        cv2.line(img=frame2, pt1=(450, 240), pt2=(450,480), color=(255, 0, 0), thickness=5, lineType=8, shift=0)
        cv2.line(img=frame2, pt1=(540, 240), pt2=(540,480), color=(255, 0, 0), thickness=5, lineType=8, shift=0)
        cv2.imshow('Capturing',frame2)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
    return frame


def SeperateTwoImage(frame):
     
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     ust = frame[0:240,:]
     altalan1 = frame[240:,0:90,:]
     altalan2 = frame[240:,90:180,:]
     altalan3 = frame[240:,180:270,:]
     altalan4 = frame[240:,270:360,:]
     altalan5 = frame[240:,360:450,:]
     altalan6 = frame[240:,450:540,:]
     altalan7 = frame[240:,540:640,:]
     return ust,altalan1,altalan2,altalan3,altalan4,altalan5,altalan6,altalan7

def preprocessing():
    frame = takePictureFromVideo()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)
    ust,altalan1,altalan2,altalan3,altalan4,altalan5,altalan6,altalan7 =  SeperateTwoImage(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ustgray = gray[60:240,:]
    altalan1gray = gray[240:420,0:90]
    altalan2gray = gray[240:420,90:180]
    altalan3gray = gray[240:420,180:270]
    altalan4gray = gray[240:420,270:360]
    altalan5gray = gray[240:420,360:450]
    altalan6gray = gray[240:420,450:540]
    altalan7gray = gray[240:420,540:640]
    
    ustblurred = cv2.GaussianBlur(ustgray,(3,3),0)  
    altalanblurred1 = cv2.GaussianBlur(altalan1gray,(3,3),0)
    altalanblurred2 = cv2.GaussianBlur(altalan2gray,(3,3),0)
    altalanblurred3 = cv2.GaussianBlur(altalan3gray,(3,3),0)
    altalanblurred4 = cv2.GaussianBlur(altalan4gray,(3,3),0)
    altalanblurred5 = cv2.GaussianBlur(altalan5gray,(3,3),0)
    altalanblurred6 = cv2.GaussianBlur(altalan6gray,(3,3),0)
    altalanblurred7 = cv2.GaussianBlur(altalan7gray,(3,3),0)
    
    ret,ustOtsuTersh = cv2.threshold(ustblurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret,altOtsuTersh1 = cv2.threshold(altalanblurred1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret,altOtsuTersh2 = cv2.threshold(altalanblurred2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret,altOtsuTersh3 = cv2.threshold(altalanblurred3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
    ret,altOtsuTersh4 = cv2.threshold(altalanblurred4,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret,altOtsuTersh5 = cv2.threshold(altalanblurred5,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret,altOtsuTersh6 = cv2.threshold(altalanblurred6,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret,altOtsuTersh7 = cv2.threshold(altalanblurred7,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return ustOtsuTersh,altOtsuTersh1,altOtsuTersh2,altOtsuTersh3,altOtsuTersh4, altOtsuTersh5, altOtsuTersh6, altOtsuTersh7,ust,altalan1,altalan2,altalan3,altalan4,altalan5,altalan6,altalan7


def edgeDetectorandFindCounter(ustHist,altalanHist1,altalanHist2,altalanHist3,altalanHist4,altalanHist5,altalanHist6,altalanHist7):
    edgesCanny = cv2.Canny(ustHist,100,200)
    edgesCanny1 =  cv2.Canny(altalanHist1,100,200)
    edgesCanny2 =  cv2.Canny(altalanHist2,100,200)
    edgesCanny3 =  cv2.Canny(altalanHist3,100,200)
    edgesCanny4 =  cv2.Canny(altalanHist4,100,200)
    edgesCanny5 =  cv2.Canny(altalanHist5,100,200)
    edgesCanny6 =  cv2.Canny(altalanHist6,100,200)
    edgesCanny7 =  cv2.Canny(altalanHist7,100,200)
    return edgesCanny, edgesCanny1, edgesCanny2, edgesCanny3, edgesCanny4, edgesCanny5, edgesCanny6, edgesCanny7



def findCorners(edgesCanny,edgesCanny1, edgesCanny2, edgesCanny3, edgesCanny4, edgesCanny5, edgesCanny6, edgesCanny7):
    corners = cv2.goodFeaturesToTrack(edgesCanny,4,0.001,10)
    corners = np.int0(corners)
    corners1 = cv2.goodFeaturesToTrack(edgesCanny1,4,0.001,10)
    corners1 = np.int0(corners1)
    corners2 = cv2.goodFeaturesToTrack(edgesCanny2,4,0.001,10)
    corners2 = np.int0(corners2)
    corners3 = cv2.goodFeaturesToTrack(edgesCanny3,4,0.001,10)
    corners3 = np.int0(corners3)
    corners4 = cv2.goodFeaturesToTrack(edgesCanny4,4,0.001,10)
    corners4 = np.int0(corners4)
    corners5 = cv2.goodFeaturesToTrack(edgesCanny5,4,0.001,10)
    corners5 = np.int0(corners5)
    corners6 = cv2.goodFeaturesToTrack(edgesCanny6,4,0.001,10)
    corners6 = np.int0(corners6)
    corners7 = cv2.goodFeaturesToTrack(edgesCanny7,4,0.001,10)
    corners7 = np.int0(corners7)
    if (np.size(corners)!=8):
        corners=np.zeros((4,2))
    if (np.size(corners1)!=8):
        corners1=np.zeros((4,2))
    if (np.size(corners2)!=8):
        corners2=np.zeros((4,2))
    if (np.size(corners3)!=8):
        corners3=np.zeros((4,2))
    if (np.size(corners4)!=8):
        corners1=np.zeros((4,2))
    if (np.size(corners5)!=8):
        corners5=np.zeros((4,2))
    if (np.size(corners6)!=8):
        corners6=np.zeros((4,2))
    if (np.size(corners7)!=8):
        corners7=np.zeros((4,2))
        
    return corners,corners1,corners2,corners3,corners4,corners5,corners6,corners7
    

def cropImages(corners,corners1,corners2,corners3,corners4,corners5,corners6,corners7,ust,altalan1,altalan2,altalan3,altalan4,altalan5,altalan6,altalan7):
    
    x = np.zeros((4,1))
    y = np.zeros((4,1))
    c=0
    for i in (0,2,4,6):
        a=corners3.item(i)
        x[c,0]=corners3.item(i)
        y[c,0]=corners3.item(i+1)
        c=c+1
    xmax = int(np.max(x))
    xmin = int(np.min(x))
    ymax = int(np.max(y))
    ymin = int(np.min(y))
    aramesafex = int((xmax-xmin)/4)
    aramesafey = int((ymax-ymin)/8)
    crop3 = altalan3[xmin+aramesafex:xmax-aramesafex, ymin+aramesafey:ymax-aramesafey]
    
    x = np.zeros((4,1))
    y = np.zeros((4,1))
    c=0
    for i in (0,2,4,6):
        a=corners2.item(i)
        x[c,0]=corners2.item(i)
        y[c,0]=corners2.item(i+1)
        c=c+1
    xmax = int(np.max(x))
    xmin = int(np.min(x))
    ymax = int(np.max(y))
    ymin = int(np.min(y))
    aramesafex = int((xmax-xmin)/4)
    aramesafey = int((ymax-ymin)/8)
    crop2 = altalan2[xmin+aramesafex:xmax-aramesafex, ymin+aramesafey:ymax-aramesafey]
    
    x = np.zeros((4,1))
    y = np.zeros((4,1))
    c=0
    for i in (0,2,4,6):
        a=corners1.item(i)
        x[c,0]=corners1.item(i)
        y[c,0]=corners1.item(i+1)
        c=c+1
    xmax = int(np.max(x))
    xmin = int(np.min(x))
    ymax = int(np.max(y))
    ymin = int(np.min(y))
    aramesafex = int((xmax-xmin)/4)
    aramesafey = int((ymax-ymin)/8)
    crop1 = altalan1[xmin+aramesafex:xmax-aramesafex, ymin+aramesafey:ymax-aramesafey]
    
    x = np.zeros((4,1))
    y = np.zeros((4,1))
    c=0
    for i in (0,2,4,6):
        a=corners4.item(i)
        x[c,0]=corners4.item(i)
        y[c,0]=corners4.item(i+1)
        c=c+1
    xmax = int(np.max(x))
    xmin = int(np.min(x))
    ymax = int(np.max(y))
    ymin = int(np.min(y))
    aramesafex = int((xmax-xmin)/4)
    aramesafey = int((ymax-ymin)/8)
    crop4 = altalan4[xmin+aramesafex:xmax-aramesafex, ymin+aramesafey:ymax-aramesafey]
    
    x = np.zeros((4,1))
    y = np.zeros((4,1))
    c=0
    for i in (0,2,4,6):
        a=corners5.item(i)
        x[c,0]=corners5.item(i)
        y[c,0]=corners5.item(i+1)
        c=c+1
    xmax = int(np.max(x))
    xmin = int(np.min(x))
    ymax = int(np.max(y))
    ymin = int(np.min(y))
    aramesafex = int((xmax-xmin)/4)
    aramesafey = int((ymax-ymin)/8)
    crop5 = altalan5[xmin+aramesafex:xmax-aramesafex, ymin+aramesafey:ymax-aramesafey]
    
    x = np.zeros((4,1))
    y = np.zeros((4,1))
    c=0
    for i in (0,2,4,6):
        a=corners6.item(i)
        x[c,0]=corners6.item(i)
        y[c,0]=corners6.item(i+1)
        c=c+1
    xmax = int(np.max(x))
    xmin = int(np.min(x))
    ymax = int(np.max(y))
    ymin = int(np.min(y))
    aramesafex = int((xmax-xmin)/4)
    aramesafey = int((ymax-ymin)/8)
    crop6 = altalan6[xmin+aramesafex:xmax-aramesafex, ymin+aramesafey:ymax-aramesafey]
    
    x = np.zeros((4,1))
    y = np.zeros((4,1))
    c=0
    
    for i in (0,2,4,6):
        a=corners.item(i)
        y[c,0]=corners.item(i)
        x[c,0]=corners.item(i+1)
        c=c+1
        
    xmax = int(np.max(x))
    xmin = int(np.min(x))
    ymax = int(np.max(y))
    ymin = int(np.min(y))
    aramesafex = int((xmax-xmin)/4)
    aramesafey = int((ymax-ymin)/8)
    crop = ust[xmin+aramesafex:xmax-aramesafex, ymin+aramesafey:ymax-aramesafey]
    
    x = np.zeros((4,1))
    y = np.zeros((4,1))
    c=0
    for i in (0,2,4,6):
        x[c,0]=corners7.item(i)
        y[c,0]=corners7.item(i+1)
        c=c+1
    xmax = int(np.max(x))
    xmin = int(np.min(x))
    ymax = int(np.max(y))
    ymin = int(np.min(y))
    aramesafex = int((xmax-xmin)/4)
    aramesafey = int((ymax-ymin)/8)
    crop7 = altalan7[xmin+aramesafex:xmax-aramesafex, ymin+aramesafey:ymax-aramesafey]
     
    
    return crop,crop1,crop2,crop3,crop4,crop5,crop6,crop7
#color Matrix 1 for  2 for red  3 for  4 for blue
color = np.zeros((8,1))

def colorfind(corners,corners1,corners2,corners3,corners4,corners5,corners6,corners7,ust,altalan1,altalan2,altalan3,altalan4,altalan5,altalan6,altalan7):
    x = np.zeros((4,1))
    y = np.zeros((4,1))
    c=0
    for i in (0,2,4,6):
        x[c,0]=corners3.item(i)
        y[c,0]=corners3.item(i+1)
        c=c+1
    xmax = int(np.max(x))
    xmin = int(np.min(x))
    ymax = int(np.max(y))
    ymin = int(np.min(y))
    crop3 = altalan3[xmin+2:xmin+5, ymin+2:ymin+5]
    point = crop3[:][2][2]
    hsv = cv2.cvtColor(crop3, cv2.COLOR_BGR2HSV)
    point = hsv[:][2][2]
    if (point[0]<20):
        color[4]=1
    if (20<point[0] and point[0]<70):
        color[4]=2
    if (70<point[0] and point[0]<155 ):
        color[4]=3
    if (155<point[0]):
        color[4]=4  
        
    
    x = np.zeros((4,1))
    y = np.zeros((4,1))
    c=0
    for i in (0,2,4,6):
        x[c,0]=corners.item(i)
        y[c,0]=corners.item(i+1)
        c=c+1
    xmax = int(np.max(x))
    xmin = int(np.min(x))
    ymax = int(np.max(y))
    ymin = int(np.min(y))
    crop = ust[xmin+2:xmin+5, ymin+2:ymin+5]
    point = crop[:][2][2]
    hsv = cv2.cvtColor(crop3, cv2.COLOR_BGR2HSV)
    point = hsv[:][2][2]
    if (point[0]<20):
        color[0]=1
    if (20<point[0] and point[0]<70):
        color[0]=2
    if (70<point[0] and point[0]<155 ):
        color[0]=3
    if (155<point[0]):
        color[0]=4   
    
    x = np.zeros((4,1))
    y = np.zeros((4,1))
    c=0 
    for i in (0,2,4,6):
        x[c,0]=corners1.item(i)
        y[c,0]=corners1.item(i+1)
        c=c+1
    xmax = int(np.max(x))
    xmin = int(np.min(x))
    ymax = int(np.max(y))
    ymin = int(np.min(y))
    crop1 = altalan1[xmin+2:xmin+5, ymin+2:ymin+5]
    point = crop1[:][2][2]
    hsv = cv2.cvtColor(crop1, cv2.COLOR_BGR2HSV)
    point = hsv[:][2][2]
    if (point[0]<20):
        color[1]=1
    if (20<point[0] and point[0]<70):
        color[1]=2
    if (70<point[0] and point[0]<155 ):
        color[1]=3
    if (155<point[0]):
        color[1]=4    
    
    
    x = np.zeros((4,1))
    y = np.zeros((4,1))
    c=0 
    for i in (0,2,4,6):
        x[c,0]=corners2.item(i)
        y[c,0]=corners2.item(i+1)
        c=c+1
        
    xmax = int(np.max(x))
    xmin = int(np.min(x))
    ymax = int(np.max(y))
    ymin = int(np.min(y))
    crop2 = altalan2[xmin+2:xmin+5, ymin+2:ymin+5]
    point = crop2[:][2][2]
    hsv = cv2.cvtColor(crop2, cv2.COLOR_BGR2HSV)
    point = hsv[:][2][2]
    if (point[0]<20):
        color[2]=1
    if (20<point[0] and point[0]<70):
        color[2]=2
    if (70<point[0] and point[0]<155 ):
        color[2]=3
    if (155<point[0]):
        color[2]=4 
        
    x = np.zeros((4,1))
    y = np.zeros((4,1))
    c=0 
    for i in (0,2,4,6):
        x[c,0]=corners4.item(i)
        y[c,0]=corners4.item(i+1)
        c=c+1
    xmax = int(np.max(x))
    xmin = int(np.min(x))
    ymax = int(np.max(y))
    ymin = int(np.min(y))
    crop4 = altalan4[xmin+2:xmin+5, ymin+2:ymin+5]
    point = crop4[:][2][2]
    hsv = cv2.cvtColor(crop4, cv2.COLOR_BGR2HSV)
    point = hsv[:][2][2]
    if (point[0]<20):
        color[4]=1
    if (20<point[0] and point[0]<70):
        color[4]=2
    if (70<point[0] and point[0]<155 ):
        color[4]=3
    if (155<point[0]):
        color[4]=4 
        
    x = np.zeros((4,1))
    y = np.zeros((4,1))
    c=0 
    for i in (0,2,4,6):
        x[c,0]=corners5.item(i)
        y[c,0]=corners5.item(i+1)
        c=c+1
    xmax = int(np.max(x))
    xmin = int(np.min(x))
    ymax = int(np.max(y))
    ymin = int(np.min(y))
    crop5 = altalan5[xmin+2:xmin+5, ymin+2:ymin+5]
    point = crop5[:][2][2]
    hsv = cv2.cvtColor(crop5, cv2.COLOR_BGR2HSV)
    point = hsv[:][2][2]
    if (point[0]<20):
        color[5]=1
    if (20<point[0] and point[0]<70):
        color[5]=2
    if (70<point[0] and point[0]<155 ):
        color[5]=3
    if (155<point[0]):
        color[5]=4 
        
        
    x = np.zeros((4,1))
    y = np.zeros((4,1))
    c=0 
    for i in (0,2,4,6):
        x[c,0]=corners6.item(i)
        y[c,0]=corners6.item(i+1)
        c=c+1
    xmax = int(np.max(x))
    xmin = int(np.min(x))
    ymax = int(np.max(y))
    ymin = int(np.min(y))
    crop6 = altalan6[xmin+2:xmin+5, ymin+2:ymin+5]
    point = crop5[:][2][2]
    hsv = cv2.cvtColor(crop6, cv2.COLOR_BGR2HSV)
    point = hsv[:][2][2]
    if (point[0]<20):
        color[6]=1
    if (20<point[0] and point[0]<70):
        color[6]=2
    if (70<point[0] and point[0]<155 ):
        color[6]=3
    if (155<point[0]):
        color[6]=4 
        
    x = np.zeros((4,1))
    y = np.zeros((4,1))
    c=0 
    for i in (0,2,4,6):
        x[c,0]=corners7.item(i)
        y[c,0]=corners7.item(i+1)
        c=c+1
    xmax = int(np.max(x))
    xmin = int(np.min(x))
    ymax = int(np.max(y))
    ymin = int(np.min(y))
    crop7 = altalan7[xmin+2:xmin+5, ymin+2:ymin+5]
    point = crop7[:][2][2]
    hsv = cv2.cvtColor(crop7, cv2.COLOR_BGR2HSV)
    point = hsv[:][2][2]
    if (point[0]<20):
        color[7]=1
    if (20<point[0] and point[0]<70):
        color[7]=2
    if (70<point[0] and point[0]<155 ):
        color[7]=3
    if (155<point[0]):
        color[7]=4
        
        
    return color
    

def cnn_model():    
    model_input = Input(shape=(32, 32, 1))
    x = BatchNormalization()(model_input)
        
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(model_input)
    x = MaxPooling2D(pool_size=(2, 2))(x) 
    
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)    
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(x)       
    x = Conv2D(64, (3, 3), activation='relu')(x)    
    x = Dropout(0.25)(x)
    
    x = Conv2D(196, (3, 3), activation='relu')(x)    
    x = Dropout(0.25)(x)
              
    x = Flatten()(x)
    
    x = Dense(512, activation='relu')(x)    
    x = Dropout(0.5)(x)
    
    y1 = Dense(11, activation='softmax')(x)
    y2 = Dense(11, activation='softmax')(x)
    y3 = Dense(11, activation='softmax')(x)
    y4 = Dense(11, activation='softmax')(x)
    y5 = Dense(11, activation='softmax')(x)
    
    model = Model(input=model_input, output=[y1, y2, y3, y4, y5])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def maxfind(pred):
    outputlayer = pred[4][0]
    maxpossibilty = np.max(outputlayer)
    prediction=0
    for i in range(10):
        a=outputlayer[i]
        if (maxpossibilty==a):
            prediction = i
    return prediction



ustHist,altalanHist1,altalanHist2,altalanHist3,altalanHist4, altalanHist5, altalanHist6, altalanHist7,ust,altalan1,altalan2,altalan3,altalan4,altalan5,altalan6,altalan7 = preprocessing()
edgesCanny,edgesCanny1, edgesCanny2, edgesCanny3, edgesCanny4, edgesCanny5, edgesCanny6, edgesCanny7 = edgeDetectorandFindCounter(ustHist,altalanHist1,altalanHist2,altalanHist3,altalanHist4,altalanHist5,altalanHist6,altalanHist7)
corners,corners1,corners2,corners3,corners4,corners5,corners6,corners7=findCorners(edgesCanny,edgesCanny1, edgesCanny2, edgesCanny3, edgesCanny4, edgesCanny5, edgesCanny6, edgesCanny7)
crop,crop1,crop2,crop3,crop4,crop5,crop6,crop7 = cropImages(corners,corners1,corners2,corners3,corners4,corners5,corners6,corners7,ust,altalan1,altalan2,altalan3,altalan4,altalan5,altalan6,altalan7)

#cropGray =   cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#crop1Gray =   cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
#crop2Gray =   cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)
crop3Gray =   cv2.cvtColor(crop3, cv2.COLOR_BGR2GRAY)
#crop4Gray =   cv2.cvtColor(crop4, cv2.COLOR_BGR2GRAY)
#crop5Gray =   cv2.cvtColor(crop5, cv2.COLOR_BGR2GRAY)
#crop6Gray =   cv2.cvtColor(crop6, cv2.COLOR_BGR2GRAY)
#crop7Gray =   cv2.cvtColor(crop7, cv2.COLOR_BGR2GRAY)


cnn_model = cnn_model()
cnn_checkpointer = ModelCheckpoint(filepath='weights.best.cnn.hdf5', 
                                   verbose=2, save_best_only=True)
cnn_model.load_weights('weights.best.cnn.hdf5')
results = np.zeros((8,1))


#cropresize=cv2.resize(cropGray,(32,32))
#pred=cnn_model.predict(cropresize.reshape(1, 32, 32, 1))
#outputlayer = pred[4]
#results[0]=maxfind(pred)

#cropresize=cv2.resize(crop1Gray,(32,32))
#pred=cnn_model.predict(cropresize.reshape(1, 32, 32, 1))
#outputlayer = pred[4]
#results[1]=maxfind(pred)

#cropresize=cv2.resize(crop2Gray,(32,32))
#pred=cnn_model.predict(cropresize.reshape(1, 32, 32, 1))
#outputlayer = pred[4]
#results[2]=maxfind(pred)


cropresize=cv2.resize(crop3Gray,(32,32))
pred=cnn_model.predict(cropresize.reshape(1, 32, 32, 1))
outputlayer = pred[4]
results[3]=maxfind(pred)
results[3]
plt.imshow(crop3Gray)
results[3]
#cropresize=cv2.resize(crop4Gray,(32,32))
#pred=cnn_model.predict(cropresize.reshape(1, 32, 32, 1))
#outputlayer = pred[4]
#results[4]=maxfind(pred)

#cropresize=cv2.resize(crop5Gray,(32,32))
#pred=cnn_model.predict(cropresize.reshape(1, 32, 32, 1))
#outputlayer = pred[4]
#results[5]=maxfind(pred)

#cropresize=cv2.resize(crop6Gray,(32,32))
#pred=cnn_model.predict(cropresize.reshape(1, 32, 32, 1))
#outputlayer = pred[4]
#results[6]=maxfind(pred)


#cropresize=cv2.resize(crop7Gray,(32,32))
#pred=cnn_model.predict(cropresize.reshape(1, 32, 32, 1))
#outputlayer = pred[4]
#results[7]=maxfind(pred)

#color=colorfind(corners,corners1,corners2,corners3,corners4,corners5,corners6,corners7,ust,altalan1,altalan2,altalan3,altalan4,altalan5,altalan6,altalan7)
