# -*- coding: utf-8 -*-
import numpy as np
import cv2
import face_recognition
from PIL import Image
from mss import mss
import os


images=[]
className=[]
encodeListKnown=None

def uploadImages():
    path = "unknown"
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        className.append(cl.split(".")[0])
        
    
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(img)[0]
        encodeList.append(encoding)
    
    return encodeList
        
        

mon={"top":120,"left":550,"height":850,"width":1200}
sct = mss()

uploadImages()
encodeListKnown = findEncodings(images)
print(className)
print("Encoding completed")

while True:
    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    
    im2 = np.array(im)
    imgS = cv2.cvtColor(im2,cv2.COLOR_BGR2RGB)
    imgS = cv2.resize(imgS,(0,0),None,0.7,0.7)

    facesCurrentFrame = face_recognition.face_locations(imgS)
    encodesCurrentFrame = face_recognition.face_encodings(imgS,facesCurrentFrame)

    for encodeFace,faceLoc in zip(encodesCurrentFrame,facesCurrentFrame):
        
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        face_distance = face_recognition.face_distance(encodeListKnown,encodeFace)
        
        matchIndex = np.argmin(face_distance)
        
        if matches[matchIndex]:
            if matchIndex == 5:
                name = "Meral Mommy"
            elif matchIndex == 6 :
                name = "Uzun Adam"
            else:
                name = className[matchIndex].upper()    
            y1,x2,y2,x1 = faceLoc
            cv2.rectangle(imgS,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(imgS,(x1,y1-35),(x2,y2),(0,255,0))
            cv2.putText(imgS,name,(x1+6,y1-6),cv2.FONT_HERSHEY_TRIPLEX,2,(255,0,0),2)
        
    
    cv2.imshow('SS',imgS)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    
            
            
            




