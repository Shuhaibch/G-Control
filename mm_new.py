import cv2
import numpy as np
import math
import pyautogui
from subprocess import call
 
cap = cv2.VideoCapture(0)

first_frame = cv2.flip(cap.read()[1],1)
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (15, 15), 0)
SCREEN_X, SCREEN_Y = pyautogui.size()
lflag = 0
flag = 0

pyautogui.FAILSAFE = False
CLICK_MESSAGE = MOVEMENT_START = None

while True:
    frame = cv2.flip(cap.read()[1],1)
    CAMERA_X, CAMERA_Y, channels = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (15, 15), 0)

    difference = cv2.absdiff(first_gray, gray_frame)
    _, thresh1 = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
    
    contours=cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    
    try: 
        cnt=max(contours,key=cv2.contourArea)
        hull = cv2.convexHull(cnt, returnPoints=False)
        drawing = np.zeros(frame.shape, np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
        used_defect = None

        for i in range(defects.shape[0]):
         s, e, f, d = defects[i, 0]
         start = tuple(cnt[s][0])
         end = tuple(cnt[e][0])
         far = tuple(cnt[f][0])
         a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
         b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
         c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
         if a>30 and b>50 and c>50:
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
         else:
            angle = 91

         if angle <= 90:
            count_defects += 1
            cv2.circle(frame, far, 5, [0, 0, 255], -1)

         cv2.line(frame, start, end, [0, 255, 0], 2)

         if count_defects == 1 and angle <= 90:
            used_defect = {"x": far[0], "y": far[1]}

        if used_defect is not None:
         best = used_defect
         if count_defects == 1:
            x = best['x']
            y = best['y']
            display_x = x
            display_y = y
            lflag = 1
            rflag = 1
            if MOVEMENT_START is not None:
                M_START = (x, y)
                x = x - MOVEMENT_START[0]
                y = y - MOVEMENT_START[1]
                x = x * (SCREEN_X / CAMERA_X)
                y = y * (SCREEN_Y / CAMERA_Y)
                MOVEMENT_START = M_START
                print("X: " + str(x) + " Y: " + str(y))
                pyautogui.moveRel(x, y)
            else:
                MOVEMENT_START = (x, y)

            cv2.circle(frame, (display_x, display_y), 5, [255, 255, 255], 20)
         elif count_defects == 2 and lflag == 1:
            pyautogui.click()
            CLICK_MESSAGE = "LEFT CLICK"
            lflag = 0
            rflag = 1
         elif count_defects == 3 and rflag == 1:
            pyautogui.rightClick()
            CLICK_MESSAGE = "RIGHT CLICK"
            rflag = 0
            lflag = 1
         elif count_defects == 4:
            CLICK_MESSAGE = "EXIT"
            lflag = 0
            rflag = 0
            cv2.destroyWindow('frame')
            cv2.destroyWindow('binary')
            cap.release()
            call("python3 g.py", shell=True)
        else:
         MOVEMENT_START = None
 
        cv2.putText(frame, CLICK_MESSAGE, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, 3)
        CLICK_MESSAGE = None
        cv2.putText(frame, "Defects: " + str(count_defects), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    except:
        pass   
    
    
    #cv2.imshow('frame',frame)
    #cv2.imshow('binary',thresh1)


    
    if cv2.waitKey(2)==ord('r'):
        print('Background reset')
        first_frame = cv2.flip(cap.read()[1],1)
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        first_gray = cv2.GaussianBlur(first_gray, (15, 15), 0)
    elif cv2.waitKey(2)==ord('e'):
        break
cv2.destroyAllWindows()
cap.release()

