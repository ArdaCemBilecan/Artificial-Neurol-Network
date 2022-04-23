import cv2
import matplotlib.pyplot as plt

# Ön hazırlık

img1 = cv2.imread("img1.JPG")
img1= cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

#normal resimleri RGB ile oluşturlmuşken opencv bize BGR verir
#Bü yüzden dönüşüm yapmak gerekir

img2 = cv2.imread("img2.JPG")
img2= cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

# 2 resmin shape'leri aynı olmak zorunda yoksa birleştiremeyiz

img1 = cv2.resize(img1,(600,600))
img2 = cv2.resize(img2,(600,600))

plt.figure()
plt.axis('off')
plt.imshow(img1)
plt.figure()
plt.axis('off')
plt.imshow(img2)

# Karıştırılmış resim = alpha*img1 + beta*img2 

blendingImg = cv2.addWeighted(src1=img1,alpha=0.5,src2=img2,beta=0.5,gamma=1)
plt.figure()
plt.axis('off')
plt.imshow(blendingImg)













