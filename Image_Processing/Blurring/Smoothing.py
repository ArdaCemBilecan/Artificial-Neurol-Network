'''
Bluring temel amacı detayı azaltır ve gürültüyü engeller

Ortalama Bulanıklaştırma:
Kuutu Filtresi kullanılır yani;
Resmin her 5 pixeli alıp bunun ortalamasını alır ve sonra 5pixele
ortalamarı yazar. Böylece bulanıklaştırmış oluruz


Gauss Bulanıklaştırma:
Ortalama ile mantık aynı kutu filtresi yerine Gauss Çekirdeği kullanılır
Pozitif ve tek olması gereken çekirdeğin genişliği ve yüksekliği
SigmaX,SigmaY ; Xve Y yönlerindeki standart sapmayı belirtmeliyiz


Medyan Bulanıklaştırma:
Çekirdek alanı altındaki tüm piksellerin medyanını alır ve merkezi
öğe bu medyan ile birleştirilir
Salt and Pepper Noises'e(Tuz ve biber gürültüsü) karşı oldukça etkili
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img),plt.axis("off"),plt.title("Orijinal"),plt.show()


# Ortalama Bulanklaştırma:
dst2 = cv2.blur(img,ksize=(3,3))
plt.figure()
plt.imshow(dst2),plt.axis("off"),plt.title("Ortalama"),plt.show()

#Gauss Bulanıklaştırma:
gaussBlur = cv2.GaussianBlur(img,ksize=(3,3),sigmaX=7)
#Y vermedik default oalrak X in değeirni alır
plt.figure()
plt.imshow(gaussBlur),plt.axis("off"),plt.title("Gaussian"),plt.show()


#Medyan Blur:
medyanBlur = cv2.medianBlur(img,ksize=3)
plt.figure()
plt.imshow(dst2),plt.axis("off"),plt.title("Median"),plt.show()

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
    

#Normalize etme
img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255 #0-1 arası olacak
GN = gaussianNoise(img)
plt.figure()
plt.imshow(GN),plt.axis("off"),plt.title("Gaussian Noise"),plt.show()


#Gauss Bulanıklaştırma :
gaussBlur2 = cv2.GaussianBlur(GN,ksize=(3,3),sigmaX=7)
#Y vermedik default oalrak X in değeirni alır
plt.figure()
plt.imshow(gaussBlur2),plt.axis("off"),plt.title("Gaussian-Normalize"),plt.show()


def saltPaperNoise(img):
    
    row,col,ch = img.shape
    s_vs_p = 0.5
    amount = 0.004
    noisy = np.copy(img)
    
    #salt beyaz 
    num_salt =int( np.ceil(amount*img.size*s_vs_p) )
    coords = [np.random.randint(0,i-1,num_salt) for i in img.shape]
    noisy[coords] = 1
    
    #paper siyah
    num_paper = int(np.ceil(amount*img.size*(1-s_vs_p)))
    coords = [np.random.randint(0,i-1,num_paper) for i in img.shape]
    noisy[coords] = 0
    
    return noisy

# Salting ekleme
spImage = saltPaperNoise(img)
plt.figure()
plt.imshow(spImage),plt.axis("off"),plt.title("Salt-paper"),plt.show()

#salting giderme
medyanBlur = cv2.medianBlur(spImage.astype(np.float32),ksize=3)
plt.figure()
plt.imshow(medyanBlur),plt.axis("off"),plt.title("Median-SP"),plt.show()


