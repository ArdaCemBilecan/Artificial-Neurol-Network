# -*- coding: utf-8 -*-
import cv2
import numpy as np


# blueLower = (84,  98,  0)
# blueUpper=(150, 255, 255)


redLower = (0,  50,  0)
redUpper=(100, 255, 255)


# Capture işlemleri

cap = cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,720)

while True:
    
    ret,frame = cap.read()
    
    if ret == True:
        # Görütü Kalitesi blurlayarak düşürelim 
        blurred = cv2.GaussianBlur(frame,(11,11),0)
        
        #HSV
        hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
        # cv2.imshow("HSV",hsv)
        
        # Mavi renk maskelemesi yapiyoruz
        mask = cv2.inRange(hsv,redLower,redUpper)
        
        
        # Görüntüdeki gürültülerden kurtulmak için erode+dilate
        mask = cv2.erode(mask,None,iterations=3)
        mask = cv2.dilate(mask,None,iterations=3)
        # cv2.imshow("Mask",mask)
        
        # Contours belirleme
        (contours,_) = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            c = max(contours,key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            ((x,y),(w,h),rotation) = rect
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            
            cv2.drawContours(frame,[box],0,(0,255,255),2)
            
            cv2.imshow("Tespit",frame)
            
            
        

    else:
        print("Hata") 
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break






