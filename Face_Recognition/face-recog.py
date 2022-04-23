import cv2
import os
import face_recognition
import numpy as np

# aces=[]
# known_names=[]

# for root, dirs, files in os.walk("./unknown"):
#     # to find the files that end with .jpg or .png and to get the face locations in every image.
#     # added all the locations and the name of the images to the dict and returned it.
#     for file in files:
#         if file.endswith(".jpg") or file.endswith(".png"):
#             face = face_recognition.load_image_file("unknown/" + file)
#             encoding = face_recognition.face_encodings(face)[0]
#             known_faces.append(encoding)
#             name = file.split(".")
#             known_names.append(name[0])
            
            
# print("PROCESSING UNKNOWN FACES")

# for file in os.listdir(UNKNOWN_FACE_DIR):
#     image = face_recognition.load_image_file("unknown/" + file)
#     img_loc = face_recognition.face_locations(image,model=MODEL)
    
#     encodings = face_recognition.face_encodings(image,img_loc)
#     image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    
#     for face_encoding , face_location in zip(encodings,img_loc):
#         result = face_recognition.compare_faces(known_faces, face_encoding,TOLERANCE)
#         match = None
#         if True in result:
#             match = known_names[result.index(True)]
#             top_left = (face_location[3],face_location[0])
#             bottom_right = (face_location[1],face_location[2])
            
#             color = [0,255,0]
            
#             cv2.rectangle(image,top_left,bottom_right,color,FRAME_THICKNESS)
            
#             top_left = (face_location[3],face_location[2])
#             bottom_right = (face_location[1],face_location[2]+22)
#             cv2.rectangle(image,top_left,bottom_right,color,cv2.FILLED)
            
#             cv2.putText(image,match,(face_location[3]+10,face_location[2]+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),FONT_THICKNESS)
#     cv2.imshow(file,image)



# known_names = []
# known_locs=[]
# def load_image_file():
#     for file in os.listdir("unknown/"):
#         face = face_recognition.load_image_file("unknown/" + file)
#         encoding = face_recognition.face_encodings(face)[0]
#         known_names.append(file.split(".")[0])
#         known_locs.append(encoding)
        

# def Recognition(images):
#     for image in images:
#         image = cv2.imread(image,1)
        
#         face_locs = face_recognition.face_locations(image,model="cnn")
#         face_encod = face_recognition.face_encodings(image,face_locs)
        
        
#         for face in face_encod:
#             match = face_recognition.compare_faces(known_locs,face)
#             name = None
#             distance = face_recognition.face_distance(known_locs,face)
            
#             min_match = np.argmin(distance)
            
#             if match[min_match]:
#                 name = known_names[min_match]
            
#             for (top, right, bottom, left), name in zip(face_locs, known_names):
#                 cv2.rectangle(image, (left - 50, top - 20), (right + 50, bottom + 20), (255, 0, 0), 2)
#                 cv2.rectangle(image, (left - 50, bottom - 15), (right + 50, bottom + 20), (255, 0, 0), cv2.FILLED)
#                 cv2.putText(image, name.upper(), (((left + right - 100 - len(name)) // 2), (bottom + 10)),
#                             cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 2)
        
#         while True:
#             cv2.imshow("Recog",image)
#             if cv2.waitKey(1) & 0xFF == ord('q'): break
        


path = "unknown"
images=[]
className=[]
myList = os.listdir(path)
print(myList)
            
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])

print(className)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    
    return encodeList


encodeListKnown = findEncodings(images)
print("Encoding Completed")


cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    imgS = cv2.resize(frame,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
 
    facesCurrentFrame = face_recognition.face_locations(imgS)
    encodesCurrentFrame = face_recognition.face_encodings(imgS,facesCurrentFrame)
    
    for encodeFace ,faceLoc in zip(encodesCurrentFrame,facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        
        face_distance = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        matchIndex = np.argmin(face_distance)
        
        if matches[matchIndex]:
            name = className[matchIndex].upper()
            
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y1-35),(x2,y2),(0,255,0))
            
            cv2.putText(frame,name,(x1+6,y1-6),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),3)
    
    cv2.imshow('Webcam',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    



        
    

        





