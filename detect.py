# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 10:29:04 2019

@author: Blaze
"""

import cv2
import datetime
#import  numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('movers.avi', fourcc, 20.0, (640,480))

ret, frame1 = cap.read()
ret, frame2 = cap.read()
#ret, rec = cap.read()

while cap.isOpened():
    
    _, img = cap.read()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    faces = face_cascade.detectMultiScale(gray1, 1.1, 4)
    datet = str(datetime.datetime.now())
    frame1 = cv2.putText(frame1, datet, (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
#    rec = cv2.putText(rec, datet, (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame1, (x,y), (x+w, y+h), (255, 0, 0), 3)
    
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        
        if cv2.contourArea(contour) < 1000:
#            out.write(frame1)
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
#        cv2.putText(frame1, "Status: ()".format('Movement'), (10, 20), font, 1, (0, 0, 255), 3)
        out.write(frame1)
#    cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
#    if contours == True:
#        out.write(frame1)

    if cv2.waitKey(40) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()