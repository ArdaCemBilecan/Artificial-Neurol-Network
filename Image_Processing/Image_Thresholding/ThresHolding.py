import cv2
import matplotlib.pyplot as plt
img = cv2.imread("img1.jpg")

# Gray Scale çevirme
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

plt.figure()
plt.axis('off')
plt.imshow(img,cmap = "gray") #color map ile griye çevirdik
plt.show()

#Thresholding
# 60-255 arasındakini eşikleri beyaz yap demek
_,thresh_img = cv2.threshold(img,thresh=60,maxval=255,type=cv2.THRESH_BINARY)
#THRESH_BINARY_INV inverse deseydik o zaman 60-255 arası siyah olacaktı diğer yerler beyaz olacaktı

plt.figure()
plt.axis('off')
plt.imshow(thresh_img,cmap = "gray") #color map ile griye çevirdik
plt.show()


# Adapting Thresholding

thresh_img2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,8)
# C sayısı ortalamadan veya ağırlıklı ortalamadan çıkartılabilecke değer = 8 dedik
# 11 block size yani siyah noktaların kalınlığı gibi düşün

plt.figure()
plt.axis('off')
plt.imshow(thresh_img2,cmap = "gray") #color map ile griye çevirdik
plt.show()
