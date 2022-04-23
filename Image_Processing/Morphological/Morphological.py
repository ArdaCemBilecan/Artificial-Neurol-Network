import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("datai_team.jpg",0) #siyah beyaz içe aktradık

plt.figure(), plt.imshow(img,cmap="gray"),plt.axis("off"),plt.title("orijinal"),plt.show()

# Erezyon --> Sınırları küçültüyoruz

kernel = np.ones((5,5),dtype=np.uint8) #kutucuğu belirtiyoruz bu kaar küçülecek anlamında

result = cv2.erode(img,kernel,iterations=1)
#iterasyon --> kaç kere erezyon yapacağımızı söylüyooruz
# plt.figure(), plt.imshow(result,cmap="gray"),plt.axis("off"),plt.title("Erode"),plt.show()


# Genişleme(dilation) --> Erezyonun tersi
result2 = cv2.dilate(img,kernel,iterations=1)
# plt.figure(), plt.imshow(result2,cmap="gray"),plt.axis("off"),plt.title("Dilate"),plt.show()

# Açılma Yöntemi (Beyaz gürültüyü önlemek için yapılır) Erezyon + Genişleme
# öncelikle beyaz gürültü ekleyelim resme


whiteNoise = np.random.randint(0,2,size=img.shape[:2])
whiteNoise = whiteNoise*255  #normalize
noise_img = whiteNoise+img
plt.figure(),plt.imshow(noise_img,cmap="gray"),plt.show()
# Acilma yapma
opening = cv2.morphologyEx(noise_img.astype(np.float32),cv2.MORPH_OPEN,kernel)
plt.figure(), plt.imshow(opening,cmap="gray"),plt.axis("off"),plt.title("Acilma"),plt.show()
'''
Öncelikle grültüdeki beyazlıkları siliyor. Sildiği için resim küçülüyor(erezyon)
sonra resmi genişletiyoruz(genişletme) ile eski halini alıyor
böylece gürültü giderilmiş oluyor
'''

#Kapatma (Siyah gürültüyü önlemek için yapılır) Genişleme+erezyon
# Öncelikle siyah gürültü ekleyelim

black_noise = np.random.randint(0,2,size=img.shape[:2])
black_noise = black_noise*(-255)
noise_img = black_noise+img
noise_img[noise_img<=-245] = 0 # filtreleme yaptık
#-245ten büyük değerleri siyahı koyuyoruz
plt.figure(),plt.imshow(noise_img,cmap="gray"),plt.title("Black"),plt.show()

closing = cv2.morphologyEx(noise_img.astype(np.float32),cv2.MORPH_CLOSE,kernel)

plt.figure(), plt.imshow(closing,cmap="gray"),plt.axis("off"),plt.title("Kapatma"),plt.show()



# Gradient = Genişleme - Erezyon ---> Kenar tespiti yapabilmek için kullanılır.

gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
plt.figure(), plt.imshow(gradient,cmap="gray"),plt.axis("off"),plt.title("Gradyan"),plt.show()











