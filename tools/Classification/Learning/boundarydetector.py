import cv2
import math
class ShapeDetector(object):
    def __init__(self,sourceImage):
        self.sourceImage = sourceImage

    def detectBoundary(self):
        grayscale = cv2.cvtColor(self.sourceImage,cv2.COLOR_RGB2GRAY)
        thresh = 255 - cv2.threshold(grayscale,grayscale.mean(),255,cv2.THRESH_BINARY)[1]
        contours,heirachy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        maxIndex = contours[0]
        maxArea = cv2.contourArea(maxIndex)
        for i in contours:
            Area = cv2.contourArea(i)
            if maxArea < Area:
                maxIndex = i
                maxArea = Area
        contour = maxIndex.reshape(-1,2)
        last = contour[0]
        index = 0
        minx = self.sourceImage.shape[1]
        cnt = 0
        for (x,y) in contour:
            #cv2.line(self.sourceImage,(last[0],last[1]),(x,y),(0,255,0))
            #cv2.circle(self.sourceImage,(x,y),1,(255,0,0),3)
            last = (x,y)
            if minx > x :
                minx = x
                index = cnt
            cnt += 1
        newPoints = []
        newPoints.extend(contour[index:])
        newPoints.extend(contour[:index])
        contour = newPoints
        cnt = 0
        (x,y) = contour[0]
        lastvalue = 0;
        chaincode = []
        length = 0
        for i in range(1,len(contour)):
            (nx,ny) = contour[i]
            dx = nx - x
            dy = ny - y
            length = math.sqrt((nx-x)**2+(ny-y)**2)
            angle = math.atan2(dy,dx) * 180 / math.pi
            angle = (angle + 360) % 360
            value = math.floor(angle / 45)
            if lastvalue != value and length > 2:
                cv2.circle(self.sourceImage,(nx,ny),1,(255,0,0),3)
                
                lastvalue = value
                chaincode.append(lastvalue)
            x,y = nx,ny
            #print(str(angle))
        #print(chaincode)
        return chaincode
