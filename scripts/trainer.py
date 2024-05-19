import os 
import cv2
import numpy as np
from PIL import Image

recog= cv2.face.LBPHFaceRecognizer_create()
path='../dataset'

def get_images(path):
    imgs_paths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    for img in imgs_paths:
        faceImg=Image.open(img).convert('L')
        faceNp=np.array(faceImg,np.uint8)
        id=int(os.path.split(img)[-1].split('.')[1]) 
        print(id)
        faces.append(faceNp)
        ids.append(id)
        cv2.imshow('train',faceNp)
        cv2.waitKey(100)
    
    return np.array(ids),faces

ids,faces =get_images(path)
    
recog.train(faces,ids)
recog.save('../recognizer/trainingData.yml')
cv2.destroyAllWindows()
    