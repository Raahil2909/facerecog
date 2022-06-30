from glob import glob
import face_recognition
import cv2
import numpy as np
import os

known_face_encodings = []
known_face_names = []

def getAllData(parPath):
    global known_face_names,known_face_encodings
    # dir struct parPath -> personName -> images
    known_face_encodings = []
    known_face_names = []
    for personName in os.listdir(parPath):
        for imgFileName in os.listdir(parPath+'/'+personName):
            imgFilePath = parPath + '/' + personName + '/' + imgFileName
            img = face_recognition.load_image_file(imgFilePath)
            imgEncs = face_recognition.face_encodings(img)
            if not imgEncs:
                continue
            imgEnc = imgEncs[0]
            known_face_encodings.append(imgEnc)
            known_face_names.append(personName)
        
    np.save('knownFaceEnc.txt',known_face_encodings)
    np.save('knownFaceNames.txt',known_face_names)
getAllData('./src1')
