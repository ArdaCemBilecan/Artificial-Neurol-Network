from GenerateModel import Generate
from DataUpload import Train_Test
from Fitting import Model_Fit
from tensorflow.keras.preprocessing.image import ImageDataGenerator



dataGen = ImageDataGenerator(width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             zoom_range = 0.1,
                             rotation_range = 10)

x_train,y_train,x_test,y_test,x_validation,y_validation = Train_Test(dataGen)

model = Generate()


hist = Model_Fit(x_train,y_train,x_test,y_test,x_validation,y_validation,model,dataGen)

results = model.evaluate(x_test, y_test, batch_size=128)

accuracy = hist.history['accuracy']
loss =hist.history['loss']
val_loss = hist.history['val_loss']
val_accuracy = hist.history['val_accuracy']

# import matplotlib.pyplot as plt
# plt.plot(accuracy)
# plt.title("Accuracy")
# plt.xlabel("Time")
# plt.ylabel("accracy")
# plt.savefig("Accuracy.png")
# plt.show()


# plt.plot(loss)
# plt.title("Loss")
# plt.xlabel("Time")s
# plt.ylabel("Loss")
# plt.savefig("Loss.png")
# plt.show()


# plt.plot(val_loss)
# plt.title("Val-Loss")
# plt.xlabel("Time")
# plt.ylabel("val_loss")
# plt.savefig("val_loss.png")
# plt.show()



# plt.plot(val_accuracy)
# plt.title("Val-Accuracy")
# plt.xlabel("Time")
# plt.ylabel("val_accuracy")
# plt.savefig("val_accuracy.png")
# plt.show()
