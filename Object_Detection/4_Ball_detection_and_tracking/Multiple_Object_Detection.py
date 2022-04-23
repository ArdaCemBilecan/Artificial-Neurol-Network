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
        
        # Contours algilama
        (contours,_) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Dış hattını bulursam bana yeterli olur zaten dışı mavi ise içi de mavidir.
        center = None
        center2= None
        if len(contours) > 0:
            boxes=[]
            Cs=[]
            rects=[]
            # 2 Adet dikdortgen istiyoruz yani 2 adet mavi cisim bulsun
            for i in range(2):
                c = max(contours,key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                Cs.append(c)
                rects.append(rect)
                #Kutucuk yapma
                box = cv2.boxPoints(rect)
                box = np.int64(box)
                boxes.append(box)
                contours.remove(c) # 2 tane en büyük bulacağımız için en büyük değeri listeden çıkarmak geerkli
                # Sonra en büyük 2. değeri bulabilelim
            # Contours Çizdir Sarı olacak
            for i in boxes:
                cv2.drawContours(frame,[i],0,(0,255,255),2)

        cv2.imshow("Tespit",frame)
        
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()      


