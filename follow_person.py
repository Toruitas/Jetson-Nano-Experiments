from adafruit_servokit import ServoKit
import cv2
print(cv2.__version__)
import numpy as np

kit = ServoKit(channels=16)

print("Loading DNN model: MobileNetSSD_deploy")

net = cv2.dnn.readNetFromCaffe("./MobileNetSSD_deploy.prototxt.txt", "./MobileNetSSD_deploy.caffemodel")

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

min_confidence = 0.9


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
    cv2.moveWindow('fg_mask_composite', 0, 410)

    # find contours of obj of interest. Array of arrays.
    # _ for unused variable
    # external for outside contour, not all for the obj
    # chain approx simple for drawing the line of the contour. Simplified x,y coords outline
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    inScaleFactor = 0.007843  # standard for mobilenet
    meanVal = 127.53  # standard for mobilenet
    expected = 300
    aspect = width / height

    resized_image = cv2.resize(frame, (round(expected * aspect), expected))

    blob = cv2.dnn.blobFromImage(resized_image, inScaleFactor, (expected, expected), meanVal, False)
    net.setInput(blob, "data")
    detections = net.forward("detection_out")

    top_detection = {
            "confidence":0,
            "x":0,
            "y":0,
            "end_x":0,
            "end_y":0,
            "idx":None,
            "label":None
        }
    detection_updated = False

    if detections.shape[2] > 0:


        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > min_confidence and confidence > top_detection['confidence']:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object

                idx = int(detections[0, 0, i, 1])

                # draw the prediction and labels only if it's a person
                if CLASSES[idx] == "person":
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    top_detection = {
                        "confidence":0,
                        "x":startX,
                        "y":startY,
                        "end_x":endX,
                        "end_y":endY,
                        "idx":idx,
                        "label":label
                    }
                    detection_updated = True

        

    # contours = sorted(detections, key=lambda x: cv2.contourArea(x), reverse=True)


    # if len(detections)>0:
    #     detection = detections[0]
    # else:
    #     continue
    # for contour in contours:
    # area = cv2.contourArea(detection)
    # (x, y, w, h) = cv2.boundingRect(detection)
    # if area >= 50:   # like 50px area
        # cv2.drawContours(frame, [contour], 0, (255,0,0), 3)
    if detection_updated:

        # It'll still try to go to the most recent position, if it loses track of the person
        cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
        cv2.putText(frame, label, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # rather than draw an unstable contour, let's find the corners of the bounding rectangle and draw that
        x = top_detection['x']
        y = top_detection['y']
        end_x = top_detection['end_x']
        end_y = top_detection['end_y']
        cv2.rectangle(frame, (x,y),(end_x,end_y), (255,0,0), 3)
        ctr_x = int((x+end_x)/2)
        ctr_y = int((y+end_y)/2)

        ctr_obj = (ctr_x, ctr_y)

        cv2.circle(frame,(ctr_x, ctr_y), 2, (255,0,0), 1)

        cv2.circle(frame,ctr_frame, 2, (255,0,0), 1)

        cv2.line(frame, (ctr_x, ctr_y),ctr_frame, (255,0,0),1)
        

        # distance from center of bounding box
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
        cv2.destroyAllWindows()
        kit.servo[1].angle = 90
        break

# cleanup program shutdown
cam.release()
cv2.destroyAllWindows()