import cv2
import matplotlib.pyplot as plt
import numpy as np


einstein = cv2.imread("einstein.jpg",0)
plt.figure(),plt.imshow(einstein,cmap="gray"),plt.axis("off"),plt.title("einstein"),plt.show()

# Cascade'ler daha önce eğitilmiş bir sınıflandırıcılardır
# Sınıflandırıcı (Yüz olup olmamasını sınıflandırıyor ) Pozitif - Negatif img
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_rect = face_cascade.detectMultiScale(einstein)

for (x,y,w,h) in face_rect:
    cv2.rectangle(einstein,(x,y),(x+w, y+h),(255,255,255),10)

plt.figure(),plt.imshow(einstein,cmap="gray"),plt.axis("off"),plt.title("Tespit"),plt.show()


# Barca

barca = cv2.imread("barcelona.jpg",0)
plt.figure(),plt.imshow(barca,cmap="gray"),plt.axis("off"),plt.title("barca"),plt.show()

face_rect = face_cascade.detectMultiScale(barca,minNeighbors=7)
#Eğer komşu rect = 7 tane değilse o zaman alma diyor yani komşusu 7> olanları alıyor

for (x,y,w,h) in face_rect:
    cv2.rectangle(barca,(x,y),(x+w, y+h),(255,255,255),10)

plt.figure(),plt.imshow(barca,cmap="gray"),plt.axis("off"),plt.title("Tespit"),plt.show()




# Kamera Face Detection

cap= cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    if ret == True:
      face_rect = face_cascade.detectMultiScale(frame,minNeighbors=7)
      for (x,y,w,h) in face_rect:
          cv2.rectangle(frame,(x,y),(x+w, y+h),(0,0,255),5)
      cv2.imshow("Face Detect",frame)
    
    if cv2.waitKey(1) &0xFF == ord('q'): break

cap.relase()
cv2.destroyAllWindows()
         














