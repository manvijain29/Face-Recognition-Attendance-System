import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

path = "IMAGES"
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
    
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
print (findEncodings(images))

def attendance(name):
    with open('ATTENDaNCE.csv','r+') as f:
        myDataList=f.readLines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            time_now=datetime.now()
            tstr=time_now.strftime('%H:%M:%S') 
            dstr=time_now.strftime('%d/%m/%Y') 
            f.writelines(f'{name},{tstr},{dstr}')  

encodeListKnown = findEncodings(images)
print("ENCODING COMPLETE!!!!")

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    imgS = cv2.resize(frame,(0,0),None,1,1)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(frame, facesCurFrame)
    
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
  
        matchIndex = np.argmin(faceDis)
        print(matchIndex)
        print(matches)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
        print(name)
        y1,x2,y2,x1=faceLoc
        y1,x2,y2,x1= y1*4,x2*4,y2*4,x1*4
        cv2.rectanle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        attendance(name)

    cv2.imshow("camera",frame)
    if cv2.waitKey(0)==13:
        break
    cap.release()
    cv2.destroyAllWindows()