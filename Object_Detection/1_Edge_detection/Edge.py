import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("london.jpg",0)

plt.figure(),plt.axis("off"),plt.imshow(img,cmap="gray"),plt.title("Orjinal"),plt.show()


edges = cv2.Canny(image=img,threshold1=0,threshold2=255,)
plt.figure(),plt.axis("off"),plt.imshow(edges,cmap="gray"),plt.title("Edges1"),plt.show()

'''
Fonksiond threshold alınırken genelde , resmin medyan değerine göre medyan hesaplanır
sonrasında aşağıdkı işleme göre low ve high bulunur ve bunlar
threshold1 = low , 2= high kabul edilir.
'''
med_val = np.median(img)
low = int(max(0, (1 -0.33)*med_val))
high = int(min(255, (1+0.33)*med_val))

print(low , high)

edges2 = cv2.Canny(image=img,threshold1=low,threshold2=high)
plt.figure(),plt.axis("off"),plt.imshow(edges2,cmap="gray"),plt.title("Edges2"),plt.show()

'''
Sudaki renk geçişleirni de kenar olarak algılıyor.
Bunu azaltmak için blurlama yapacağız ki bu tür algılamaları azaltalım
'''

blur_img = cv2.blur(img,ksize=(5,5))
med_val_blur = np.median(blur_img)
low = int(max(0, (1 -0.33)*med_val_blur))
high = int(min(255, (1+0.33)*med_val_blur))

edges3 = cv2.Canny(image=blur_img,threshold1=low,threshold2=high)
plt.figure(),plt.axis("off"),plt.imshow(edges3,cmap="gray"),plt.title("Blur"),plt.show()
