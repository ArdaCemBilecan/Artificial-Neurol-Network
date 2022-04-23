'''
Kontur tespiti, aynı renk veya yoğunluğa sahip tüm kesintisiz noktaları
birleştirmeyi amaçlayan yöntemdir

Bu işlem şekil analizi ve nesne tanıma için kullanılır
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("contour.jpg",0)
contours, hierarch = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#2.parametre  = Hem internal hem external (iç ve dış konturları) bulmak istiyorum demek
#3.parametre = Yatay,dikey ve çapraz bölümleri sıkıştırmayı sağlar. Yalnızca uç noktaları bırakır

external_contours = np.zeros(img.shape)
internal_contours = np.zeros(img.shape)

for i in range(len(contours)):
    if hierarch[0][i][3] == -1:    # eğer 0 ve i ve 2. paratmere 0 ise bu external demek
        cv2.drawContours(external_contours,contours,i,255,-1) # 255 renk , -1 dediğimiz kalınlık
        # -1 yazdığımız zaman konturlarla sınırlanan alanı doldur demek
    else:
        cv2.drawContours(internal_contours,contours,i,255,-1)


# Resimdeki beyaz şekilleri alıyor yani
plt.figure(),plt.imshow(external_contours,cmap="gray"),plt.axis("off"),plt.title("External"),plt.show()

# Resimdeki siyah şekilleri alıyor 
plt.figure(),plt.imshow(internal_contours,cmap="gray"),plt.axis("off"),plt.title("Internal"),plt.show()        
        
    
