import cv2
import numpy as np
from collections import deque

# Nesnelerimizin merkezini depolamak için kullanacağımız deque
buffer_size = 16
pts = deque(maxlen = buffer_size)

# Mavi için HSV kod aralıklarını belirtiyor.
blueLower = (84,  98,  0)
blueUpper=(179, 255, 255)

#capture islemi

cap = cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)

# Kamere acik oldugu surece
while True:
    ret,frame = cap.read()
    if ret == True:
        #Blurlayarak kalite dusuruyoruz
        blurred = cv2.GaussianBlur(frame,(11,11),0)
        # frame'leri HSV formatına çevirmek gerekiyor
        hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
        # cv2.imshow("HSV",hsv)
        
        # Mavi renk için maskeleme işlemi. Mavi rengi tanitiyoruz
        mask = cv2.inRange(hsv,blueLower,blueUpper)
        # cv2.imshow("Maskeli Hali",mask)
        
        # Maskeli halinde bazi gürültüler var onlardan kurtulmak lazım
        # Onun için erezyon+genişleme yapmak gerekir.
        mask = cv2.erode(mask,None,iterations=3)
        mask = cv2.dilate(mask,None,iterations=3)
        cv2.imshow("Erode+Dilate",mask)
        
        # Contours algilama
        (contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Dış hattını bulursam bana yeterli olur zaten dışı mavi ise içi de mavidir.
        center = None
        
        if len(contours) > 0:
            # En buyuk contoursu alalım
            c = max(contours,key=cv2.contourArea)
            #Dikdortgen cizdirecegiz
            rect = cv2.minAreaRect(c)
            ((x,y),(width,height),rotation) = rect
            
            s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x),np.round(y),np.round(width),np.round(height),np.round(rotation))
            print(s)
            
            #Kutucuk yapma
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            
            # Moment --> Görüntünün merkezi bulmaya yarayan yapı
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"]/M["m00"]))
            #  x,y noklatını bulduk
            
            # Contours Çizdir Sarı olacak
            cv2.drawContours(frame,[box],0,(0,255,255),2)
            
            # Merkeze 1 tane nokta çizme
            cv2.circle(frame,center,5,(255,0,255),-1)
            
            #bilgileri ekrana koy
            cv2.putText(frame, s, (25,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
            
        
        # deque
        pts.appendleft(center)
        #Nesne takibi için
        for i in range(1, len(pts)):
            
            if pts[i-1] is None or pts[i] is None: continue
        
            cv2.line(frame, pts[i-1], pts[i],(0,255,0),3) 
            
            cv2.imshow("Takip Etmeli",frame)
        
    if cv2.waitKey(1) & 0xFF == ord("q"): break
        


