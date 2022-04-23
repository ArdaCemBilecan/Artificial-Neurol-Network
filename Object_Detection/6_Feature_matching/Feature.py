import cv2
import matplotlib.pyplot as plt

'''
Bu yöntem birden fazla nesne tespiti için değil tek bir nesne tespti yapar

Birden çok kişinin olduğu bir resimde seni bulur ve yüzünü gösterir.

Brute-Force eşleştiricisi , bir görüntüdeki bir özelliğin tanımlayıcısı başka bir görüntünün
diğer tüm özellikleri ile eşleştirir ve mesafeye göre sonuç döndürür.

*** Tüm özellikleri kontrol ettiği için yavaş çalışır.
'''


chos = cv2.imread("chocolates.jpg", 0)
# plt.figure(), plt.imshow(chos, cmap = "gray"),plt.axis("off")

# aranacak olan görüntü
cho = cv2.imread("nestle.jpg", 0)
# plt.figure(), plt.imshow(cho, cmap = "gray"),plt.axis("off")

# orb tanımlayıcı
# köşe-kenar gbi nesneye ait özellikler
orb = cv2.ORB_create()

# anahtar nokta tespiti
kp1, des1 = orb.detectAndCompute(cho, None)
kp2, des2 = orb.detectAndCompute(chos, None)

# Brute-Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
# noktaları eşleştir
matches = bf.match(des1, des2)

# mesafeye göre sırala
matches = sorted(matches, key = lambda x: x.distance)

# eşleşen resimleri görselleştirelim
plt.figure()
img_match = cv2.drawMatches(cho, kp1, chos, kp2, matches[:20], None, flags = 2)
plt.imshow(img_match), plt.axis("off"),plt.title("orb")

# ORB ile iyi sonuç çıkmadı çünkü ana resimde nestle çikolata hem yatay değil hem bazı farklılıklar var
# Sift Tanımlaycısı ile deneyelim 
# opencv' de olmasına karşın dışarıdan eklenmiştir o yüzden Sift yüklü değilse
# pip install opencv-contrib-python --user yazınız.

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

# Anahtar nokta tespiti sift ile

kp1, des1 = sift.detectAndCompute(cho, None) # None = MAskeleme olmayacak
kp2, des2 = sift.detectAndCompute(chos, None)

matches = bf.knnMatch(des1, des2, k = 2) # KNN en yakın komşu eşleşmesi
# 2 tane eşleşme yapıcaz

best = []
# m --> short of match
for m1,m2 in matches:
    
    if m1.distance < 0.75*m2.distance:
        best.append([m1])
          
plt.figure()
sift_matches = cv2.drawMatchesKnn(cho,kp1,chos,kp2,best,None, flags = 2)
plt.imshow(sift_matches), plt.axis("off"), plt.title("sift")






