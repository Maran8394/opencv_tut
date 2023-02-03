import cv2
import random
import numpy as np


class OpenCVTut:
    def changeImagePixels(self):
        img = cv2.imread("assets/apple.jpg")
        gray1 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        # NOTE 1: Trying to change the Pixels
        for i in range(100):
            for j in range(img.shape[1]):
                img[i][j] = random.randint(0,255)
        
        cv2.imshow("Apple", gray1)
        d = cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def copyPixels(self):
        img = cv2.imread("assets/apple.jpg")
        # NOTE 2:Trying to copy of Pixels
        sli = img[10:50,240:260]
        img[30:70, 340:360]=sli 

        cv2.imshow("Apple", img)
        d = cv2.waitKey(0)
        cv2.destroyAllWindows()

    def videoMirroring(self):
        cap = cv2.VideoCapture("assets/vtest.avi")
        while cap.isOpened():
            ret, frame = cap.read()
            height = int(cap.get(4))
            width = int(cap.get(3))

            image = np.zeros(frame.shape, np.uint8)
            smaller_frame1 = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
            smaller_frame2 = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
            smaller_frame3 = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
            smaller_frame4 = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
            image[:height//2,:width//2]=smaller_frame1
            image[:height//2,width//2:]=smaller_frame2
            image[height//2:,:width//2]=smaller_frame3
            image[height//2:,width//2:]=smaller_frame4

            cv2.imshow("Video", image)
            if cv2.waitKey(1) == ord("q"):
                break
        cap.release()

    def colorDetectionInVideo(self):
        # NOTE 5: Color and Color Detection
        # NOTE -> HSV -> Hue Saturation and Lightness/Brightness
        cap = cv2.VideoCapture("assets/vtest.avi")
        while cap.isOpened():
            ret, frame = cap.read()
            height = int(cap.get(4))
            width = int(cap.get(3))

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_red = np.array([0, 100, 20])
            upper_red = np.array([179,255,255])

            mask = cv2.inRange(hsv,lower_red, upper_red)
            res = cv2.bitwise_and(frame,frame, mask=mask)
            
            cv2.imshow("Videeeo", mask)
            cv2.imshow("Video", res)
            if cv2.waitKey(1) == ord("q"):
                break
        cap.release()

    def drawInVideo(self):
        # NOTE 4: Drawing
        cap = cv2.VideoCapture("assets/vtest.avi")
        while cap.isOpened():
            ret, frame = cap.read()
            height = int(cap.get(4))
            width = int(cap.get(3))

            img = cv2.line(frame,(0,0),(width,height),(250,0,0),5,)
            img = cv2.rectangle(frame,(100,100),(200,200),(200,0,0),-1)
            img2 = cv2.circle(img,(250,250),30,(200,0,0),-1)

            font = cv2.FONT_HERSHEY_COMPLEX
            img = cv2.putText(img,"Maran",(200, height-10),font,1,(0,0,0),2)
            cv2.imshow("Video", img2)
            if cv2.waitKey(1) == ord("q"):
                break
        cap.release()

    def findCorners(self):
        img = cv2.imread("assets/chessboard1.png")
        img = cv2.resize(img, (0,0), fx=0.5,fy=0.5)
        gray1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray1,100, 0.1,10)

        # floating to int
        corners = np.int0(corners)

        for corner in corners:
            x,y = np.ravel(corner) # ravel - to flatten array
            cv2.circle(img, (x,y),5,(255,0,0),-1)

        for i in range(len(corners))    :
            for j in range(i+1,len(corners)):
                corner1 = tuple(corners[i][0])
                corner2 = tuple(corners[j][0])
                color =tuple(map(lambda x:int(x),np.random.randint(0,255, size=3)))
                cv2.line(img, corner1,corner2, color,3 )

        cv2.imshow( "CORNERS", img)
        d = cv2.waitKey(0)
        cv2.destroyAllWindows()
        


o = OpenCVTut()
o.findCorners()
