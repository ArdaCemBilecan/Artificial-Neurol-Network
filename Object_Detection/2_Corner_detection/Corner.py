import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
Tanımı: Şu nki konumla bir sonraki konum arasındaki yoğunluk farkı fazla ise
o zaman köşeden geçmişim demektir.
'''
img = cv2.imread("sudoku.jpg",0)
img = np.float32(img)

plt.figure(),plt.imshow(img,cmap="gray"),plt.axis("off"),plt.show()

# Harris Corner Detection

dst = cv2.cornerHarris(img,blockSize=2,ksize=3,k=0.04)
# blockSize = komşuluk yani kaç komşusuna bakayım demek
# Kernel bildiğimiz gibi kutucuk boyutu
# k ise harris denklemindeki free parametre
plt.figure(),plt.imshow(dst,cmap="gray"),plt.axis("off"),plt.title("Harris"),plt.show()

# Koşeleri dilate ile genişleteliml ki daha belirgin olsun

dst = cv2.dilate(dst,None)
img[dst>0.2*dst.max()] = 1
plt.figure(),plt.imshow(dst,cmap="gray"),plt.axis("off"),plt.title("Dilate"),plt.show()

# Shi Tomsai Detection

img2 = cv2.imread("sudoku.jpg", 0)
img2 = np.float32(img2)
corners = cv2.goodFeaturesToTrack(img2, 120, 0.01, 10)
#120 = Kaç kenar bulsun onu diyoruz
#0.01 free parametre
# 10 ise köşeler arasındaki uzaklık 
corners = np.int64(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img2, (x,y),3,(125,125,125),cv2.FILLED)
    
plt.imshow(img2)
plt.axis("off")




