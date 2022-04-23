from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical



def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img


def Train_Test(dataGen):
    path = '../Classification_Fruits/Data'
    files = os.listdir(path)
    X = []
    Y = []
    
    for i in files:
        myPath = path+'/'+i
        myFile = os.listdir(myPath)
        for file in myFile:
            newPath = myPath+'/'+file
            images = os.listdir(newPath)
            for j in images:
                fullPath = newPath+'/'+j
                img = cv2.imread(fullPath)
                img = cv2.resize(img,(32,32))
                X.append(img)
                Y.append(files.index(i))
                
    
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25, random_state=0)
    x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=0.25, random_state=42)

    dataGen.fit(x_train)
    x_train = np.array(list(map(preProcessing, x_train)))
    x_test = np.array(list(map(preProcessing, x_test)))
    x_validation = np.array(list(map(preProcessing, x_validation)))
    
    x_train = x_train.reshape(-1,32,32,1)
    
    x_test = x_test.reshape(-1,32,32,1)
    x_validation = x_validation.reshape(-1,32,32,1)
    
    y_train = to_categorical(y_train, 24)
    y_test = to_categorical(y_test, 24)
    y_validation = to_categorical(y_validation, 24)

    return x_train,y_train,x_test,y_test,x_validation,y_validation
