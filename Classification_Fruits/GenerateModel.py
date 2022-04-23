from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten


def Generate():
    model = Sequential()
    
    model.add(Conv2D(input_shape = (32,32,1), filters = 32, kernel_size = (5,5), activation = "relu", padding = "same"))
    model.add(Conv2D(32,kernel_size=(3,3) , activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64,kernel_size=(3,3) , activation='relu', padding='same'))
    model.add(Conv2D(64,kernel_size=(3,3) , activation='relu', padding='same'))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64,kernel_size=(3,3) , activation='relu', padding='same'))
    model.add(Conv2D(64,kernel_size=(3,3) , activation='relu', padding='same'))
    
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(24,activation='softmax'))
    
    model.compile(loss = "categorical_crossentropy", optimizer=("Adam"), metrics = ["accuracy"])
    
    return model