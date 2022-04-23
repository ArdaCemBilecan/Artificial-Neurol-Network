import cv2
import os

files = os.listdir()

img_list = []

for f in files:
    if f.endswith(".jpg"):
        img_list.append(f)

# HOG tanımlayıcısı

hog = cv2.HOGDescriptor()
# SVM Ekle
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


for img in img_list:
    image = cv2.imread(img)
    (rects,weights) = hog.detectMultiScale(image,padding = (8,8),scale=1.05)
    
    for (x,y,w,h) in  rects:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    
    cv2.imshow("Yaya",image)
    
    if cv2.waitKey(0) &0xFF == ord('q'):
        continue
    
    
    
    
    
    
    
    
    
    
    
    