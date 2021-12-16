# Integrated code
# @author: LTTS Machine Vision

import time
import cv2 as cv
import numpy as np

start_time = time.time()
cell1 = cv.imread("image_1.png",0)
ret,thresh_binary = cv.threshold(cell1,150, 255,cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(image =thresh_binary , mode = cv.RETR_TREE,method = cv.CHAIN_APPROX_SIMPLE)
mask = np.zeros(cell1.shape[:2],dtype=np.uint8)

for i,cnt in enumerate(contours):
        # if the contour has no other contours inside of it
        if hierarchy[0][i][2] == -1 :
            
            if  cv.contourArea(cnt) <= 5050 and cv.contourArea(cnt) >= 1000:
                cv.drawContours(mask,[cnt], 0, (255), -1)
                
cv.imwrite('Testing_image.png', mask)
image = cv.imread('Testing_image.png')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (3,3), 0)
thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
cnts = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
a=0
z=0
print("Units: cm")
for c in cnts:
    perimeter = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.04 * perimeter, True)

    if len(approx) > 1: 
        x,y,w,h = cv.boundingRect(c)
        diameter = w/185
        M = cv.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv.drawContours(image,[c], 0, (36,255,12), 4)
        cv.circle(image, (cX, cY), 15, (320, 159, 22), -1) 
        a=a+1
        cv.line(image, (x, y + int(h/2)), (x + w, y + int(h/2)), (156, 188, 24), 1)
        cv.putText(image, "D:{}".format(a), (cX - 20, cY - 6), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print( "D"+ str(a) + ": ","{:.2f}".format(diameter))
        
cv.imwrite('image.png', image)
cv.imwrite('thresh.png', thresh)
cv.imwrite('opening.png', opening)
cv.imshow("Final Image.png", image)
print("--- %s seconds ---" % (time.time() - start_time))

cv.waitKey(0)
cv.destroyAllWindows()

