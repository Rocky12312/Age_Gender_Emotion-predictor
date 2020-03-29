import cv2
import numpy as np
import sys
import os
import time
import cv2
import pafy
import numpy as np
import pandas as pd
import keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model


a = []#for storing gender_labels
b = []#for storing male and female ages
c = []#for storing emotions
d = []

classifier_gender = load_model('model.h888')
#male age predictor
classifier_male = load_model('model.h3333')
#female age predictor
classifier_female = load_model('model.h444444')
#emotion predictor
classifier_emotion = load_model('model.h55')

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_detector(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    d.append(len(faces))
    if faces is ():
        return (0,0,0,0), np.zeros((100,100), np.uint8), img
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
    try:
        roi_gray = cv2.resize(roi_gray, (100,100), interpolation = cv2.INTER_AREA)
    except:
        return (x,w,y,h), np.zeros((100,100), np.uint8), img
    return (x,w,y,h), roi_gray, img


def face_detector1(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    if faces is ():
        return (0,0,0,0), np.zeros((48,48), np.uint8), img
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
    try:
        roi_gray = cv2.resize(roi_gray, (48,48), interpolation = cv2.INTER_AREA)
    except:
        return (x,w,y,h), np.zeros((48,48), np.uint8), img
    return (x,w,y,h), roi_gray, img



def getting_video_feed(x):
    Total_frame_with_person_count = 0
    frame_count = 0
    cap = cv2.VideoCapture(x)
    while True:
        frame_count = frame_count+1
        ret, frame = cap.read()
        rect, face, image = face_detector(frame)
        rect1, face1, image1 = face_detector1(frame)
        if np.sum([face]) != 0.0:
            roi = face.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
        
            roi1 = face1.astype("float") / 255.0
            roi1 = img_to_array(roi1)
            roi1 = np.expand_dims(roi1, axis=0)
            
            class_labels_gender = {0:"fl",1:"ml"}
            preds_gender = classifier_gender.predict(roi)[0]
            label_gender = class_labels_gender[preds_gender.argmax()]  
            label_position_gender = (rect[0] + int((rect[1]/2)), rect[2] + 25)
            cv2.putText(image, label_gender, label_position_gender , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
            a.append(label_gender)
           
            Total_frame_with_person_count=Total_frame_with_person_count+1
            if label_gender == "ml":
                #class_labels_male = validation_generator_male.class_indices
                class_labels_male = {0:"mm(1-9)",1:"mm(10-14)",2:"mm(15-18)",3:"mm(19-28)",4:"mm(28-40)",5:"mm(45-59)",6:"mm(60-x)"}
                preds_male = classifier_male.predict(roi)[0]
                label_male = class_labels_male[preds_male.argmax()]
                b.append(label_male)
                
                #class_labels_emotion = validation_generator_emotion.class_indices
                class_labels_emotion = {0:"Angry",1:"Fear",2:"Happy",3:"Neutral",4:"Sad",5:"Surprise"}
                preds_emotion = classifier_emotion.predict(roi1)[0]
                label_emotion = class_labels_emotion[preds_emotion.argmax()]
                c.append(label_emotion)
                
            else:
                class_labels_female = {0:"fm(1-9)",1:"fm(10-14)",2:"fm(15-18)",3:"fm(19-28)",4:"fm(28-40)",5:"fm(45-59)",6:"fm(60-x)"}
                preds_female = classifier_female.predict(roi)[0]
                label_female = class_labels_female[preds_female.argmax()]
                b.append(label_female)
                
                #class_labels_emotion = validation_generator_emotion.class_indices
                class_labels_emotion = {0:"Angry",1:"Fear",2:"Happy",3:"Neutral",4:"Sad",5:"Surprise"}
                preds_emotion = classifier_emotion.predict(roi1)[0]
                label_emotion = class_labels_emotion[preds_emotion.argmax()]
                c.append(label_emotion)
                
            
        """else:
            cv2.putText(image, "No Face Found", (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)"""
        
        #cv2.imshow('All', image)
        if(frame_count>100):
            break
        ch = cv2.waitKey(25)
        """if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break"""

    count_male = 0
    count_female = 0
    m1happy9,m10happy14,m15happy18,m19happy28,m28happy40,m45happy59,m60happyx = 0,0,0,0,0,0,0
    m1sad9,m10sad14,m15sad18,m19sad28,m28sad40,m45sad59,m60sadx = 0,0,0,0,0,0,0
    m1fear9,m10fear14,m15fear18,m19fear28,m28fear40,m45fear59,m60fearx = 0,0,0,0,0,0,0
    m1surprise9,m10surprise14,m15surprise18,m19surprise28,m28surprise40,m45surprise59,m60surprisex = 0,0,0,0,0,0,0
    m1neutral9,m10neutral14,m15neutral18,m19neutral28,m28neutral40,m45neutral59,m60neutralx = 0,0,0,0,0,0,0
    m1angry9,m10angry14,m15angry18,m19angry28,m28angry40,m45angry59,m60angryx = 0,0,0,0,0,0,0
    f1happy9,f10happy14,f15happy18,f19happy28,f28happy40,f45happy59,f60happyx = 0,0,0,0,0,0,0
    f1sad9,f10sad14,f15sad18,f19sad28,f28sad40,f45sad59,f60sadx = 0,0,0,0,0,0,0
    f1fear9,f10fear14,f15fear18,f19fear28,f28fear40,f45fear59,f60fearx = 0,0,0,0,0,0,0
    f1surprise9,f10surprise14,f15surprise18,f19surprise28,f28surprise40,f45surprise59,f60surprisex = 0,0,0,0,0,0,0
    f1neutral9,f10neutral14,f15neutral18,f19neutral28,f28neutral40,f45neutral59,f60neutralx = 0,0,0,0,0,0,0
    f1angry9,f10angry14,f15angry18,f19angry28,f28angry40,f45angry59,f60angryx = 0,0,0,0,0,0,0
    for (i,j,k) in zip(a,b,c):
        if i=="ml" and j=="mm(1-9)" and k =="Happy":
            m1happy9 = m1happy9+1
        if i=="ml" and j=="mm(10-14)" and k =="Happy":
            m10happy14 = m10happy14+1
        if i=="ml" and j=="mm(15-18)" and k =="Happy":
            m15happy18 = m15happy18+1
        if i=="ml" and j=="mm(19-28)" and k =="Happy":
            m19happy28 = m19happy28+1
        if i=="ml" and j=="mm(28-40)" and k =="Happy":
            m28happy40 = m28happy40+1
        if i=="ml" and j=="mm(45-59)" and k =="Happy":
            m45happy59 = m45happy59+1    
        if i=="ml" and j=="mm(60-x)" and k =="Happy":
            m60happyx = m60happyx+1
        if i=="ml" and j=="mm(1-9)" and k =="Sad":
            m1sad9 = m1sad9+1
        if i=="ml" and j=="mm(10-14)" and k =="Sad":
            m10sad14 = m10sad14+1
        if i=="ml" and j=="mm(15-18)" and k =="Sad":
            m1sad18 = m15sad18+1
        if i=="ml" and j=="mm(19-28)" and k =="Sad":
            m19sad28 = m19sad28+1
        if i=="ml" and j=="mm(28-40)" and k =="Sad":
            m28sad40 = m28sad40+1
        if i=="ml" and j=="mm(45-59)" and k =="Sad":
            m45sad59 = m45sad59+1    
        if i=="ml" and j=="mm(60-x)" and k =="Sad":
            m60sadx = m60sadx+1
        if i=="ml" and j=="mm(1-9)" and k =="Neutral":
            m1neutral9 = m1neutral9+1
        if i=="ml" and j=="mm(10-14)" and k =="Neutral":
            m10neutral14 = m10neutral14+1
        if i=="ml" and j=="mm(15-18)" and k =="Neutral":
            m15neutral18 = m15neutral18+1
        if i=="ml" and j=="mm(19-28)" and k =="Neutral":
            m19neutral28 = m19neutral28+1
        if i=="ml" and j=="mm(28-40)" and k =="Neutral":
            m28neutral40 = m28neutral40+1
        if i=="ml" and j=="mm(45-59)" and k =="Neutral":
            m45neutral59 = m45neutral59+1    
        if i=="ml" and j=="mm(60-x)" and k =="Neutral":
            m60neutralx = m60neutralx+1
        if i=="ml" and j=="mm(1-9)" and k =="Fear":
            m1fear9 = m1fear9+1
        if i=="ml" and j=="mm(10-14)" and k =="Fear":
            m10fear14 = m10fear14+1
        if i=="ml" and j=="mm(15-18)" and k =="Fear":
            m1fear18 = m15fear18+1
        if i=="ml" and j=="mm(19-28)" and k =="Fear":
            m19fear28 = m19fear28+1
        if i=="ml" and j=="mm(28-40)" and k =="Fear":
            m28fear40 = m28fear40+1
        if i=="ml" and j=="mm(45-59)" and k =="Fear":
            m45fear59 = m45fear59+1    
        if i=="ml" and j=="mm(60-x)" and k =="Fear":
            m60fearx = m60fearx+1
        if i=="ml" and j=="mm(1-9)" and k =="Surprise":
            m1surprise9 = m1surprise9+1
        if i=="ml" and j=="mm(10-14)" and k =="Surprise":
            m10surprise14 = m10surprise14+1
        if i=="ml" and j=="mm(15-18)" and k =="Surprise":
            m15surprise18 = m15surprise18+1
        if i=="ml" and j=="mm(19-28)" and k =="Surprise":
            m19surprise28 = m19surprise28+1
        if i=="ml" and j=="mm(28-40)" and k =="Surprise":
            m28surprise40 = m28surprise40+1
        if i=="ml" and j=="mm(45-59)" and k =="Surprise":
            m45surprise59 = m45surprise59+1    
        if i=="ml" and j=="mm(60-x)" and k =="Surprise":
            m60surprisex = m60surprisex+1
        if i=="ml" and j=="mm(1-9)" and k =="Angry":
            m1angry9 = m1angry9+1
        if i=="ml" and j=="mm(10-14)" and k =="Angry":
            m10angry14 = m10angry14+1
        if i=="ml" and j=="mm(15-18)" and k =="Angry":
            m1angry18 = m15angry18+1
        if i=="ml" and j=="mm(19-28)" and k =="Angry":
            m19angry28 = m19angry28+1
        if i=="ml" and j=="mm(28-40)" and k =="Angry":
            m28angry40 = m28angry40+1
        if i=="ml" and j=="mm(45-59)" and k =="Angry":
            m45angry59 = m45angry9+1    
        if i=="ml" and j=="mm(60-x)" and k =="Angry":
            m60angryx = m60angryx+1
        if i=="fl" and j=="fm(1-9)" and k =="Happy":
            f1happy9 = f1happy9+1
        if i=="fl" and j=="fm(10-14)" and k =="Happy":
            f10happy14 = f10happy14+1
        if i=="fl" and j=="fm(15-18)" and k =="Happy":
            f15happy18 = f15happy18+1
        if i=="fl" and j=="fm(19-28)" and k =="Happy":
            f19happy28 = f19happy28+1
        if i=="fl" and j=="fm(28-40)" and k =="Happy":
            f28happy40 = f28happy40+1
        if i=="fl" and j=="fm(45-59)" and k =="Happy":
            f45happy59 = f45happy59+1    
        if i=="fl" and j=="fm(60-x)" and k =="Happy":
            f60happyx = f60happyx+1
        if i=="fl" and j=="fm(1-9)" and k =="Sad":
            f1sad9 = f1sad9+1
        if i=="fl" and j=="fm(10-14)" and k =="Sad":
            f10sad14 = f10sad14+1
        if i=="fl" and j=="fm(15-18)" and k =="Sad":
            f1sad18 = f15sad18+1
        if i=="fl" and j=="fm(19-28)" and k =="Sad":
            f19sad28 = f19sad28+1
        if i=="fl" and j=="fm(28-40)" and k =="Sad":
            f28sad40 = f28sad40+1
        if i=="fl" and j=="fm(45-59)" and k =="Sad":
            f45sad59 = f45sad59+1    
        if i=="fl" and j=="fm(60-x)" and k =="Sad":
            f60sadx = f60sadx+1
        if i=="fl" and j=="fm(1-9)" and k =="Neutral":
            f1neutral9 = f1neutral9+1
        if i=="fl" and j=="fm(10-14)" and k =="Neutral":
            f10neutral14 = f10neutral14+1
        if i=="fl" and j=="fm(15-18)" and k =="Neutral":
            f15neutral18 = f15neutral18+1
        if i=="fl" and j=="fm(19-28)" and k =="Neutral":
            f19neutral28 = f19neutral28+1
        if i=="fl" and j=="fm(28-40)" and k =="Neutral":
            f28neutral40 = f28neutral40+1
        if i=="fl" and j=="fm(45-59)" and k =="Neutral":
            f45neutral59 = f45neutral59+1    
        if i=="fl" and j=="fm(60-x)" and k =="Neutral":
            f60neutralx = f60neutralx+1
        if i=="fl" and j=="fm(1-9)" and k =="Fear":
            f1fear9 = f1fear9+1
        if i=="fl" and j=="fm(10-14)" and k =="Fear":
            f10fear14 = f10fear14+1
        if i=="fl" and j=="fm(15-18)" and k =="Fear":
            f1fear18 = f15fear18+1
        if i=="fl" and j=="fm(19-28)" and k =="Fear":
            f19fear28 = f19fear28+1
        if i=="fl" and j=="fm(28-40)" and k =="Fear":
            f28fear40 = f28fear40+1
        if i=="fl" and j=="fm(45-59)" and k =="Fear":
            f45fear59 = f45fear59+1    
        if i=="fl" and j=="fm(60-x)" and k =="Fear":
            f60fearx = f60fearx+1
        if i=="fl" and j=="fm(1-9)" and k =="Surprise":
            f1surprise9 = f1surprise9+1
        if i=="fl" and j=="fm(10-14)" and k =="Surprise":
            f10surprise14 = f10surprise14+1
        if i=="fl" and j=="fm(15-18)" and k =="Surprise":
            f15surprise18 = f15surprise18+1
        if i=="fl" and j=="fm(19-28)" and k =="Surprise":
            f19surprise28 = f19surprise28+1
        if i=="fl" and j=="fm(28-40)" and k =="Surprise":
            f28surprise40 = f28surprise40+1
        if i=="fl" and j=="fm(45-59)" and k =="Surprise":
            f45surprise59 = f45surprise59+1    
        if i=="fl" and j=="fm(60-x)" and k =="Surprise":
            f60surprisex = f60surprisex+1
        if i=="fl" and j=="fm(1-9)" and k =="Angry":
            f1angry9 = f1angry9+1
        if i=="fl" and j=="fm(10-14)" and k =="Angry":
            f10angry14 = f10angry14+1
        if i=="fl" and j=="fm(15-18)" and k =="Angry":
            m1angry18 = m15angry18+1
        if i=="fl" and j=="fm(19-28)" and k =="Angry":
            f19angry28 = f19angry28+1
        if i=="fl" and j=="fm(28-40)" and k =="Angry":
            f28angry40 = f28angry40+1
        if i=="fl" and j=="fm(45-59)" and k =="Angry":
            f45angry59 = f45angry9+1    
        if i=="fl" and j=="fm(60-x)" and k =="Angry":
            f60angryx = f60angryx+1         
            
    print(frame_count)
    print(m1sad9)
    a.clear()
    b.clear()
    c.clear()
    d.clear()
    cap.release()
    cv2.destroyAllWindows()
