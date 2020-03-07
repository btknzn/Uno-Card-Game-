import numpy as np
import cv2
from matplotlib import pyplot as plt

def takePictureFromVideo():
    video = cv2.VideoCapture(0)
    while True:
        check,frame = video.read()
        cv2.imshow('Capturing',frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
    return frame


def SeperateTwoImage(frame):
     ust = frame[0:240,:,:]
     altalan1 = frame[240:,0:160,:]
     altalan2 = frame[240:,160:320,:]
     altalan3 = frame[240:,320:480,:]
     altalan4 = frame[240:,480:640,:]
     return ust,altalan1,altalan2,altalan3,altalan4

def preprocessing():
    frame = takePictureFromVideo()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)
    ust,altalan1,altalan2,altalan3,altalan4 =  SeperateTwoImage(frame)
    ustgray = cv2.cvtColor(ust,cv2.COLOR_BGR2GRAY)
    altalan1gray = cv2.cvtColor(altalan1,cv2.COLOR_BGR2GRAY)
    altalan2gray = cv2.cvtColor(altalan2,cv2.COLOR_BGR2GRAY)
    altalan3gray = cv2.cvtColor(altalan3,cv2.COLOR_BGR2GRAY)
    altalan4gray = cv2.cvtColor(altalan4,cv2.COLOR_BGR2GRAY)
    ustblurred = cv2.GaussianBlur(ustgray,(5,5),0)  
    altalanblurred1 = cv2.GaussianBlur(altalan1gray,(5,5),0)
    altalanblurred2 = cv2.GaussianBlur(altalan2gray,(5,5),0)
    altalanblurred3 = cv2.GaussianBlur(altalan3gray,(5,5),0)
    altalanblurred4 = cv2.GaussianBlur(altalan4gray,(5,5),0)
    plt.imshow(ustblurred)
    ustHist = cv2.equalizeHist(ustblurred)
    altalanHist1 = cv2.equalizeHist(altalan1gray)
    altalanHist2 = cv2.equalizeHist(altalan2gray)
    altalanHist3 = cv2.equalizeHist(altalan3gray)    
    altalanHist4 = cv2.equalizeHist(altalan4gray)
    plt.imshow(ustHist)
    return ustHist,altalanHist1,altalanHist2,altalanHist3,altalanHist4
def edgeDetectorandFindCounter(ustHist,altalanHist1,altalanHist2,altalanHist3,altalanHist4):
    
    
ustHist,altalanHist1,altalanHist2,altalanHist3,altalanHist4= preprocessing() 
edgeDetectorandFindCounter(ustHist,altalanHist1,altalanHist2,altalanHist3,altalanHist4)   