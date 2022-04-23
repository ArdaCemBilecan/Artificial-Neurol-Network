#Resmi içeriye aktarma ve yeni oluşturma

import cv2

img = cv2.imread("messi5.png")

cv2.imshow("First Img",img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

k = cv2.waitKey(0) &0xff

if k == 27: # esc
    cv2.destroyAllWindows()
elif k == ord('s'):
    #yeni resim dosyası oluşturma
    cv2.imwrite("messi_gray.png",img)
    cv2.destroyAllWindows()

