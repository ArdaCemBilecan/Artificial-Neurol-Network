import cv2
import numpy as np
img = cv2.imread("lenna.png")
cv2.imshow("Orijinal",img)

hor = np.hstack((img,img)) #numpy ile 2 resmin matirini birleştirdik

cv2.imshow("Horizontol",hor)

# dikey

ver = np.vstack((img,img))

cv2.imshow("Vertical",ver)