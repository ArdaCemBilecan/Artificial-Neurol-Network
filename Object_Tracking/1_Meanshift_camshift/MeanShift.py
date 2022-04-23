import cv2
import numpy as np

# Camera

cap = cv2.VideoCapture(0)
cap.set(3,720)
cap.set(4,480)


ret,frame = cap.read()

if ret == False:
    print("Error")

#Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_rects = face_cascade.detectMultiScale(frame)

(face_x,face_y,w,h) = tuple(face_rects[0])

track_window=(face_x,face_y,w,h) #meanshift algoritma girdisi

# Region of interest

roi = frame[face_y:face_y+h,face_x:face_x+w] # Yüzün tespiti
hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

histogram = cv2.calcHist([hsv_roi],[0],None,[180],[0,180]) # takip için histogram gereklidir.

cv2.normalize(histogram,histogram,0,255,cv2.NORM_MINMAX)


# takip için gerkeli durdurma kritlerleri
# count = hesaplanacak maksimum öge sayısı
# eps = Degisiklik

term_crit = (cv2.TERM_CRITERIA_EPS or cv2.TERM_CRITERIA_COUNT,5,1)

while True:
    ret,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        
        # Histogramı bir görüntüde bulmak için kullanıyoruz. eşleme yapılıyor
        # Takip yapılıyor
        #Pixel karşılaştırma
        dst = cv2.calcBackProject([hsv],[0],histogram,[0,180],1)
        
        
        ret,track_window= cv2.meanShift(dst,track_window,term_crit)
        
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)
        
        cv2.imshow("Takip",img2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):break

cap.release()
cv2.destroyAllWindows()
        
        
        
 














