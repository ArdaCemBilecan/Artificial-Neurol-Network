# opencv,numpy,matplotlib kütüphanelerini içe aktaralım

import cv2
import numpy as np
import matplotlib.pyplot as plt
# resmi siyah beyaz olarak içe aktaralım resmi çizdirelim

img = cv2.imread("odev2.jpg",0)
# resim üzerinde bulunan kenarları tespit edelim ve görselleştirelim edge detection


med_val = np.median(img)
low = int(max(0, (1 -0.33)*med_val))
high = int(min(255, (1+0.33)*med_val))
edge = cv2.Canny(img,threshold1 = low , threshold2 = high)          
plt.figure(),plt.axis("off"),plt.imshow(edge,cmap="gray"),plt.title("Edges"),plt.show()


# yüz tespiti için gerekli haar cascade'i içe aktaralım
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# yüz tespiti yapıp sonuçları görselleştirelim
face_rects = face_cascade.detectMultiScale(img,scaleFactor=1.045,minNeighbors=7)

for (x,y,w,h) in face_rects:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)   
plt.figure(),plt.axis("off"),plt.imshow(img,cmap="gray"),plt.title("Tespit"),plt.show()


# HOG ilklendirelim insan tespiti algoritmamızı çağıralım ve svm'i set edelim
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# resme insan tespiti algoritmamızı uygulayalım ve görselleştirelim
(rects,weights) = hog.detectMultiScale(img,padding = (8,8),scale=1.05)
for (x,y,w,h) in face_rects:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)

plt.figure(),plt.axis("off"),plt.imshow(img,cmap="gray"),plt.title("HOG"),plt.show()