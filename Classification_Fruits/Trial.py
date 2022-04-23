import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss
from GenerateModel import Generate

mon={"top":120,"left":550,"height":850,"width":1200}
sct = mss()

model = Generate()

model.save_weights("weights.h5")

labels = ["Apple", "Banana", "Cherry","Corn","Eggplant","Grape","Kiwi","Lemon","Mandarine","Mango","Onion","Orange",
          "Peach","Pear","Pepper","Pinapple","Plum","Pomegranate","Pomelo Sweetie","Potato","Strawberry","Tomato",
          "Walnut","Watermelone"]




while True:
    
    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    im2 = np.array(im.convert("L").resize((850, 1200)))
    image = im2
    image = np.asarray(image)
    image = cv2.resize(image, (32,32))
    image = image.reshape(1,32,32,1)
    classIndex = int(model.predict_classes(image))
    
    cv2.putText(im2,labels[classIndex],(50,50),cv2.FONT_HERSHEY_DUPLEX, 1,(0,255,255),3)
    cv2.imshow("img",im2)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
