'''
Kedinin yüzünü tespit etmek için şu yöntem kullanılır;
Bir kutu oluşturulur ve bu kutu resim üstünde önce yukardan başlayarak
soldan sağa her yeri gezer. Sonra aşağı iner.
Bulmak istediğimiz kısma(cat_face) en yakın matrix değerini bulur ve
oraya bize gösterir
'''
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("cat.jpg",0)

templete=cv2.imread("cat_face.jpg",0)

h,w = templete.shape

# 2 resim arasında korelasyonu çıkartan methodlar

methods = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED',
           'cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']


for meth in methods:
    method = eval(meth) # 'cv2.TM_CCOEFF' ---> cv2.TM_CCOEFF  stringten çıkarıyor
    res = cv2.matchTemplate(img,templete,method)
    print(res.shape) # orijinal resimle aynı olmak zorunda
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0]+w , top_left[1]+h)
    cv2.rectangle(img,top_left,bottom_right,255,3)
    
    plt.figure()
    plt.subplot(121), plt.imshow(res, cmap = "gray")
    plt.title("Eşleşen Sonuç"), plt.axis("off")
    plt.subplot(122), plt.imshow(img, cmap = "gray")
    plt.title("Tespit edilen Sonuç"), plt.axis("off")
    plt.suptitle(meth)
    
    plt.show()
    
    
    
    
    
    
    
    
    
    