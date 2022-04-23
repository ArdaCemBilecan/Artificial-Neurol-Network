import cv2
import matplotlib.pyplot as plt
import numpy as np
# resmi siyah beyaz olarak içe aktaralım
img = cv2.imread("odev1.jpg",0)
img2 = cv2.imread("odev1.jpg") # renkli olanı
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
# resmi çizdirelim
plt.figure(),plt.axis("off"),plt.imshow(img,cmap="gray"),plt.title("Siyah-Beyaz"),plt.show()
plt.figure(),plt.axis("off"),plt.imshow(img2),plt.title("Orijinal"),plt.show()

# resmin boyutuna bakalım
w,h = img.shape

# resmi 4/5 oranında yeniden boyutlandıralım ve resmi çizdirelim
w= int( w*(4/5) )
h= int (h*(4/5) )
resize_img = cv2.resize(img2,(w,h))
plt.figure(),plt.axis("off"),plt.imshow(resize_img,cmap="gray"),plt.title("resize_img"),plt.show()

# orijinal resme bir yazı ekleyelim mesela "kopek" ve resmi çizdirelim
text_img = cv2.putText(img2,"Kedi",(664,208),cv2.FONT_HERSHEY_TRIPLEX,2,(255,0,0))
plt.figure(),plt.axis("off"),plt.imshow(text_img,cmap="gray"),plt.title("text_img"),plt.show()

# orijinal resmin 50 threshold değeri üzerindekileri beyaz yap altındakileri siyah yapalım, 
# binary threshold yöntemi kullanalım ve resmi çizdirelim
_,thresh_img =cv2.threshold(img2,thresh=50,maxval=255,type=cv2.THRESH_BINARY)
plt.figure(),plt.axis("off"),plt.imshow(thresh_img),plt.title("Threshold"),plt.show()

_,inv = cv2.threshold(img2,thresh=50,maxval=255,type=cv2.THRESH_BINARY_INV)
plt.figure(),plt.axis("off"),plt.imshow(inv),plt.title("Inverse"),plt.show()


# orijinal resme gaussian bulanıklaştırma uygulayalım ve resmi çizdirelim

gausBlur = cv2.GaussianBlur(img2,ksize=(3,3),sigmaX=7)
plt.figure(),plt.axis("off"),plt.imshow(gausBlur),plt.title("Gauss"),plt.show()

# orijinal resme Laplacian  gradyan uygulayalım ve resmi çizdirelim

laplacian = cv2.Laplacian(img2,ddepth=cv2.CV_64F)
plt.figure(),plt.axis("off"),plt.imshow(laplacian),plt.title("Laplacian"),plt.show()

#Blending ile kenar kenar bulma
sobelx = cv2.Sobel(img2,ddepth=cv2.CV_16S,dx=1,dy=0,ksize=5)
sobely= cv2.Sobel(img2,ddepth=cv2.CV_16S,dx=0,dy=1,ksize=5)
blending = cv2.addWeighted(src1=sobelx,alpha=0.5,src2=sobely,beta=0.5,gamma=1)
plt.figure(),plt.axis("off"),plt.imshow(blending),plt.title("Blending"),plt.show()

# orijinal resmin histogramını çizdirelim
color = ("b", "g", "r")
plt.figure()
for i,c in enumerate(color):
    hist = cv2.calcHist([img2],channels=[i],mask=None,histSize=[256],ranges=[0,256])
    plt.plot(hist, color = c)


img_hist = cv2.calcHist([img2], channels = [0], mask = None, histSize = [256], ranges = [0,256])
plt.figure()
plt.plot(img_hist)


