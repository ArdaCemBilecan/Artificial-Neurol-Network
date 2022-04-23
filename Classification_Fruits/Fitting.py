from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf

np.random.seed(1)

def Model_Fit(x_train,y_train,x_test,y_test,x_validation,y_validation,model,dataGen):
    
    batch_size = 64
    
    hist = model.fit_generator(dataGen.flow(x_train, y_train, batch_size = batch_size), 
                                        validation_data = (x_validation, y_validation),
                                        epochs = 10,steps_per_epoch = x_train.shape[0]//batch_size, shuffle = 1)
    
    model.save_weights("weights.h5")
    return hist
    
