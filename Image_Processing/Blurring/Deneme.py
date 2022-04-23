import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(),plt.axis("off"),plt.title("Orijinal"),plt.imshow(img),plt.show()

# Siyah beyaz noktalar ekleme

row,col,ch = img.shape
s_vs_p = 0.5
amount = 0.004
noisy = np.copy(img)

num_alt = int(np.ceil(amount*s_vs_p*img.size)) 

coords =[np.random.randint(0,i-1,num_alt) for i in img.shape]

noisy[coords] = -255

num_paper = int(np.ceil(amount*img.size*(1-s_vs_p)))
coords = [np.random.randint(0,i-1,num_paper) for i in img.shape]
noisy[coords] = 255

plt.figure(),plt.axis("off"),plt.title("Nosiy"),plt.imshow(noisy),plt.show()

def gaussianNoise(img):
    #gürültü elde etme
    row,col,ch = img.shape  # ch dediği chanlle yani RGB mi BGR mı onu söyler
    mean = 0
    varians = 0.05
    sigma = varians**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = img + gauss
    return noisy
    