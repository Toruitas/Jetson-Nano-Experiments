from adafruit_servokit import ServoKit
import cv2
print(cv2.__version__)
import numpy as np

kit = ServoKit(channels=16)

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

disp_w = 640
disp_h = 480
flip = 2  # a param on rpi cam. Else comes out upside down

pan = 90  # center
tilt = 90  # center
margin = 15


kit.servo[0].angle = pan
kit.servo[1].angle = tilt


# this launches gstreamer on the camera 
# 21 fps
# opencv likes bgr
cam_set ='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(disp_w)+', height='+str(disp_h)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'

cam = cv2.VideoCapture(cam_set)
width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
half_width = int(width/2)
half_height = int(height/2)
ctr_frame = (half_width, half_height)

print(width, height, half_width, half_height)

# cam = cv2.VideoCapture(0)  # 0 or 1 for a webcam. Likely cam 0.

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
        ctr_x = int(x+w/2)
        ctr_y = int(y+h/2)

        ctr_obj = (ctr_x, ctr_y)

        cv2.circle(frame,(ctr_x, ctr_y), 2, (255,0,0), 1)

        cv2.circle(frame,ctr_frame, 2, (255,0,0), 1)

        cv2.line(frame, (ctr_x, ctr_y),ctr_frame, (255,0,0),1)
        

        # distance
        error = np.linalg.norm(np.array(ctr_obj) - np.array(ctr_frame))
        margin = 10

        if error > 50:
            step = 1
        else:
            step = 0.1

        if error > margin:
            # adjust
            if ctr_x > half_width:
                pan = pan - step
            if ctr_x < half_width:
                pan = pan + step
            if ctr_y > half_height:
                tilt = tilt + step
            if ctr_y < half_height:
                tilt = tilt - step

        if pan > 180:
            pan = 180
        
        if pan < 0:
            pan = 0

        if tilt > 180:
            tilt = 180

        if tilt < 0:
            tilt = 0

        

        # print(f"{pan}, {tilt}")

        kit.servo[0].angle = pan
        kit.servo[1].angle = tilt
            
    # cv2.drawContours(frame, contours, 0, (255,0,0), 3)  # -1 draws all contours, 0 first in array. Contours are random order, so not like most improtant is first

    cv2.imshow('nano_cam', frame)
    cv2.moveWindow('nano_cam', 0, 0)

    # gotta exit gracefully
    if cv2.waitKey(1) == ord('q'): # sees if a key is pressed every 1 ms. If it is q, quit
        kit.servo[0].angle = 90
        kit.servo[1].angle = 90
        break

# cleanup program shutdown
cam.release()
cv2.destroyAllWindows()