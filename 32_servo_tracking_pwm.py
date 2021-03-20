import Jetson.GPIO as GPIO
import cv2
print(cv2.__version__)
import numpy as np

def nothing(x):
    """trackbar always wants a fn called, so need a dummy"""
    pass

# build track bars
cv2.namedWindow('Trackbars')
cv2.moveWindow('Trackbars', 1320, 0)
cv2.createTrackbar('hue-low', 'Trackbars', 137, 179, nothing)  # need 6 trackbars. Low, high values for hue, saturation, and value.
cv2.createTrackbar('hue-high', 'Trackbars', 179, 179, nothing)  # name, window to put in, initial value, max value, callback fn
cv2.createTrackbar('hue2-low', 'Trackbars', 0, 179, nothing)  # need 6 trackbars. Low, high values for hue, saturation, and value.
cv2.createTrackbar('hue2-high', 'Trackbars', 14, 179, nothing)  # name, window to put in, initial value, max value, callback fn
cv2.createTrackbar('sat-low', 'Trackbars', 140, 255, nothing)  
cv2.createTrackbar('sat-high', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('val-low', 'Trackbars',111, 255, nothing)
cv2.createTrackbar('val-high', 'Trackbars',255, 255, nothing)

GPIO.setmode(GPIO.BOARD)

GPIO.setup((32,33), GPIO.OUT)

pwm_pan = GPIO.PWM(32, 50)  # 50hz 
pwm_tilt = GPIO.PWM(33, 50)  # 50hz

pwm_pan.start(0)
pwm_tilt.start(0)

disp_w = 320*2
disp_h = 240*2
flip = 2  # a param on rpi cam. Else comes out upside down

pan = 0  # center
tilt = 0  # center
margin = 15


def set_angle(angle:int, pwm):
    duty = 7.5 + 2.5/45 * angle
    pwm.ChangeDutyCycle(duty)

# this launches gstreamer on the camera 
# 21 fps
# opencv likes bgr
cam_set ='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(disp_w)+', height='+str(disp_h)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'

cam = cv2.VideoCapture(cam_set)
width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

# cam = cv2.VideoCapture(0)  # 0 or 1 for a webcam. Likely cam 0.

set_angle(pan, pwm_pan)
set_angle(tilt, pwm_tilt)

# we want to read frames in a while loop
while True:
    ret, frame = cam.read()
    # frame = cv2.imread('img/smarties.png')
    

    # img is in rgb, but we want hsv. So convert it.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    hue_low = cv2.getTrackbarPos('hue-low','Trackbars')
    hue_high = cv2.getTrackbarPos('hue-high','Trackbars')
    # we need another hue mask since at the 0,1 there's a discontinuity
    # we make this and add it so we can get both parts of the red
    hue2_low = cv2.getTrackbarPos('hue2-low','Trackbars')
    hue2_high = cv2.getTrackbarPos('hue2-high','Trackbars')

    sat_low = cv2.getTrackbarPos('sat-low','Trackbars')
    sat_high = cv2.getTrackbarPos('sat-high','Trackbars')

    value_low = cv2.getTrackbarPos('val-low','Trackbars')
    value_high = cv2.getTrackbarPos('val-high','Trackbars')

    # create bound arrays
    l_b = np.array([hue_low, sat_low, value_low])
    u_b = np.array([hue_high, sat_high, value_high])
    l_b2 = np.array([hue2_low, sat_low, value_low])
    u_b2 = np.array([hue2_high, sat_high, value_high])

    # foreground mask, just keep the colors we want within the range l_b to u_b. All will be 1 or 0. Black or white mask.
    fg_mask = cv2.inRange(hsv, l_b, u_b)
    fg_mask2 = cv2.inRange(hsv, l_b2, u_b2)
    fg_mask_composite = cv2.add(fg_mask, fg_mask2)
    cv2.imshow('fg_mask_composite', fg_mask_composite)
    cv2.moveWindow('fg_mask_composite', 0, 410)

    # find contours of obj of interest. Array of arrays.
    # _ for unused variable
    # external for outside contour, not all for the obj
    # chain approx simple for drawing the line of the contour. Simplified x,y coords outline
    contours, hierarchy = cv2.findContours(fg_mask_composite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    if len(contours)>0:
        contour = contours[0]
    else:
        continue
    # for contour in contours:
    area = cv2.contourArea(contour)
    (x, y, w, h) = cv2.boundingRect(contour)
    if area >= 50:   # like 50px area
        # cv2.drawContours(frame, [contour], 0, (255,0,0), 3)
        # rather than draw an unstable contour, let's find the corners of the bounding rectangle and draw that
        cv2.rectangle(frame, (x,y),(x+w,y+h), (255,0,0), 3)
        ctr_x = x+w/2
        ctr_y = y+h/2

        # calculate diff between ctr of obj and ctr of screen
        error_pan = ctr_x - width/2
        error_tilt = ctr_y - height/2

        # if 


        # create a box which the 
        if abs(w/2)> abs(h/2):
            margin = abs(w/2)
        else:
            margin = abs(h/2)
        print("Error", error_pan, error_tilt, margin)
        
        if abs(error_pan) > margin:
            not_pan = pan + error_pan / (width/180)  # attempt to scale pixels to degrees
            print(not_pan)

        if abs(error_tilt) > margin:
            not_tilt = tilt + error_tilt / (height/180)
            print(not_tilt)

        # adjust
        if error_pan > margin :
            pan = pan - 1
        if error_pan < margin:
            pan = pan + 1
        if error_tilt > margin:
            tilt = tilt + 1
        if error_tilt < margin:
            tilt = tilt - 1

        if pan > 90:
            pan = 90
        
        if pan < -90:
            pan = -90

        if tilt > 90:
            tilt = 90

        if tilt < -90:
            tilt = -90

        

        print(f"{pan}, {tilt}")

        set_angle(pan, pwm_pan)
        set_angle(tilt, pwm_tilt)
            
    # cv2.drawContours(frame, contours, 0, (255,0,0), 3)  # -1 draws all contours, 0 first in array. Contours are random order, so not like most improtant is first

    cv2.imshow('nano_cam', frame)
    cv2.moveWindow('nano_cam', 0, 0)

    # gotta exit gracefully
    if cv2.waitKey(1) == ord('q'): # sees if a key is pressed every 1 ms. If it is q, quit
        set_angle(0, pwm_pan)
        set_angle(0, pwm_tilt)
        break

# cleanup program shutdown
cam.release()
cv2.destroyAllWindows()