import cv2
import numpy as np 
import sqlite3

faceDetect= cv2.CascadeClassifier('../haarcascade_frontalface_default.xml') 
cam=cv2.VideoCapture(0)

def InsertOrUpdate(id, name, age):  #Dealong with sqlite database
    conn=sqlite3.connect('../database.db') #connect to database
    cmd="SELECT * FROM STUDENTS WHERE ID ="+str(id)
    data= conn.execute(cmd)
    isRecoedExist=0
    for row in data:
        isRecoedExist=1
    if (1 == isRecoedExist):
        conn.execute('UPDATE STUDENTS SET Name=? WHERE id=?', (name,id,))
        conn.execute('UPDATE STUDENTS SET Name=? WHERE id=?', (age,id,))
    else: 
        conn.execute("INSERT INTO STUDENTS (id,name,age) values(?,?,? )",(id,name,age))
    conn.commit()
    conn.close()


# insert user info
id=input("Enter the student's ID : ")
name=input("Enter the student's Name : ")
age=input( "Enter the student's Age : ")

InsertOrUpdate(id, name, age)

#detecting Face in web cam
sampleNum=0
 
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        sampleNum +=1 
        cv2.imwrite('../dataset/user.'+str(id)+"."+str(sampleNum)+'.jpg', gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100)
    cv2.imshow('face', img)
    cv2.waitKey(1)
    if (sampleNum>20):
        break

cam.release()
cv2.destroyAllWindows()
