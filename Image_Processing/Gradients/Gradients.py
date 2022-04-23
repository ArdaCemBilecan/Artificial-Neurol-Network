import cv2
import matplotlib.pyplot as plt

img = cv2.imread("sudoku.jpg",0)
plt.figure(),plt.axis("off"),plt.imshow(img,cmap="gray"),plt.show()


# x gradyan

sobelx = cv2.Sobel(img,ddepth=cv2.CV_16S,dx=1,dy=0,ksize=5)
plt.figure(),plt.axis("off"),plt.imshow(sobelx,cmap="gray"),plt.title("SobelX"),plt.show()

# y gradyan
sobely = cv2.Sobel(img,ddepth=cv2.CV_16S,dx=0,dy=1,ksize=5)
plt.figure(),plt.axis("off"),plt.imshow(sobely,cmap="gray"),plt.title("SobelY"),plt.show()

#Laplacian gradian---> Hem x hem y gradyan bulur
laplacian = cv2.Laplacian(img,ddepth=cv2.CV_16S)
plt.figure(),plt.axis("off"),plt.imshow(laplacian,cmap="gray"),plt.title("laplacian"),plt.show()

blendingImg = cv2.addWeighted(src1=sobelx,alpha=0.5,src2=sobely,beta=0.5,gamma=1)
plt.figure(),plt.axis("off"),plt.imshow(blendingImg,cmap="gray"),plt.title("blendingImg"),plt.show()
