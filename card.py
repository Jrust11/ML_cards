import numpy as np
import cv2
import imutils
import math

cap = cv2.VideoCapture(0)

img = cv2.imread('card.jpg',0)
img = cv2.GaussianBlur(img, (7, 7), 0)

per_low = 1000000000000
the_card_is = 0

suits = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']


for x in suits:

    img2 = cv2.imread(f'{x}.jpg',0)
    #img2 = cv2.GaussianBlur(img2, (7, 7), 0)

    ##############################################
    width = 250
    height = 350

    corners = np.float32([[316,49],[419,157],[53,128],[143,250]])
    correct_corners = np.float32([[0,0],[width,0],[0,height],[width,height]])


    matrix = cv2.getPerspectiveTransform(corners,correct_corners)
    result = cv2.warpPerspective(img,matrix,(width,height))
    crop = result[0:100, 0:38]
    crop = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    rank = crop[0:55, 5:38]
    #suit = crop[60:100, 0:38]

    rank = cv2.resize(rank, (250,350))

    ##############################################


    #img2 = cv2.resize(img2, (250,350))

    overlap = cv2.absdiff(rank,img2)

    per = int(np.sum(overlap)/255)

    if per < (per_low):
        per_low = per
        print(per_low)
        the_card_is = x


print(f'The Card is a {the_card_is}')

while(True):


    # Capture frame-by-frame
    ret, frame = cap.read()
    img = cv2.imread('card.jpg')

    corners = np.array([[316,49],[419,157],[53,128],[143,250]])

    for x in range (0,4):
        cv2.circle(img,(corners[x][0],corners[x][1]),5,(0,0,255),cv2.FILLED)


    a = int(np.linalg.norm(corners[0][0] - corners[3][0])/2)
    b = int(np.linalg.norm(corners[0][1] - corners[3][1])/2)

    cv2.putText(img, text='Testing', org=(a,b),fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=1, lineType=cv2.LINE_AA)


    ##############################################

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.namedWindow("Image")
    cv2.imshow('Image',img)

    cv2.namedWindow("Overlap")
    cv2.imshow('Overlap',overlap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()
