import numpy as np
import cv2
import imutils
import math
from time import sleep


suits = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']

cap = cv2.VideoCapture(0)


def canny(img, low_threshold, high_threshold):

    kernel = np.ones((5,5))

    #imgCanny = cv2.Canny(img, 50, 100)
    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((10,10)))
    #img = cv2.dilate(img,kernel,iterations=1)

    return cv2.Canny(img, low_threshold, high_threshold)





while(True):

    ret, frame = cap.read()

    frame = cv2.GaussianBlur(frame, (11, 11), 0)

    frame = canny(frame,5,100)

    #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    thresh=cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 20)

    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    for cnt in contours:

        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)

        x,y,w,h=cv2.boundingRect(cnt)

        if abs(w-h) <= 4:
            frame = cv2.drawContours(frame,[cnt],0,(0,0,255),-1)


    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
