import numpy as np
import cv2
import os 
import sqlite3 

faceDetect=cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
cam= cv2.VideoCapture(0)

recog= cv2.face.LBPHFaceRecognizer_create()
recog.read('../recognizer/trainingData.yml')

def getProfileId(id):
    conn=sqlite3.connect('../database.db')
    data= conn.execute('SELECT * FROM students WHERE id=?',(id,))
    profile=None
    for row in data:
        profile = row
    conn.close()
    return profile

while (True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces= faceDetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf=recog.predict(gray[y:y+h,x:x+w])
        profile=getProfileId(id)
        print(profile)
        if(profile!=None):
            cv2.putText(img,'Name:'+ str(profile[0]), (x,y+h+20), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,127),3)
            cv2.putText(img,'Age:'+ str(profile[2]), (x,y+h+45), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,127),3)
            # cv2.putText(img,'Name:'+ str(profile[1]), (x,y+h+20), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,127),3)
    
    cv2.imshow("face",img)
    if(cv2.waitKey(1)==ord('q')):
        break
cam.release()
cv2.destroyAllWindows()

