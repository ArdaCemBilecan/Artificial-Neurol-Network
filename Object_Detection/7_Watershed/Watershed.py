import cv2
import matplotlib.pyplot as plt
import numpy as np

#İçe aktarma
coin = cv2.imread("coins.jpg")

# Paraların üstündeki detayları azaltmak için blurring işlemi yapıyoruz Median

coin_blur= cv2.medianBlur(coin,13)
plt.figure(),plt.imshow(coin_blur),plt.axis("off"),plt.title("Median"),plt.show()

# Siyah beyaza çevirme
coin_blur = cv2.cvtColor(coin_blur,cv2.COLOR_BGR2GRAY)

# Paraları bulmak için threshold yapıyoruz
_,coin_threshold = cv2.threshold(coin_blur,65,255,cv2.THRESH_BINARY)
plt.figure(),plt.imshow(coin_threshold,cmap="gray"),plt.axis("off"),plt.title("Threshold"),plt.show()
#içeride bazı beyazlıklar kalıyor ama sorun değil çünkü external contours yapacağız

# Resimler arasında köprüler var yani temaslar var. Bunu ortadan kaldırmak için Erezyon+Genişleme yapmak lazım
# yani Açılma yapacğaız çünkü beyazlıkları azaltmak gerekir

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(coin_threshold,cv2.MORPH_OPEN,kernel,iterations = 2)
plt.figure(),plt.imshow(opening,cmap="gray"),plt.axis("off"),plt.title("Opening"),plt.show()

# Nesneler arası distance bulma

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5) # Öklid distance

plt.figure(),plt.imshow(dist_transform,cmap="gray"),plt.axis("off"),plt.title("Distance"),plt.show()


# Nesneleri küçültelim 
ret,sure_foreground = cv2.threshold(dist_transform,0.4*np.max(dist_transform),255,0)
plt.figure(),plt.imshow(sure_foreground,cmap="gray"),plt.axis("off"),plt.title("Foreground"),plt.show()
# Resmi küçülttüğümüz için adacıkları bulmuş olduk

# Arka plan için nesneleri büyüt

sure_background = cv2.dilate(opening,kernel,1)
sure_foreground = np.uint8(sure_foreground)

# Arka plan - önplan yapacağız böylece nesneler ayrıt edilebilir olacak
unknown = cv2.subtract(sure_background,sure_foreground) 
plt.figure(),plt.imshow(unknown,cmap="gray"),plt.axis("off"),plt.title("unknown"),plt.show()
# Bağlantı bulma

ret,marker  = cv2.connectedComponents(sure_foreground)
marker = marker+1
marker[unknown == 255] = 0


# Watershed
marker = cv2.watershed(coin,marker)
plt.figure(),plt.imshow(marker,cmap="gray"),plt.axis("off"),plt.title("Marker"),plt.show()

# Contours belrileme
contours, hierarchy = cv2.findContours(marker.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(coin, contours,i,(255,0,0),5)
plt.figure(),plt.imshow(coin),plt.axis("off")
















