import numpy as np                        # fundamental package for scientific computing
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import cv2

print("Environment Ready")

# loading the model
net = cv2.dnn.readNetFromCaffe("./MobileNetSSD_deploy.prototxt.txt", "./MobileNetSSD_deploy.caffemodel")

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
DEPTH_W = 1280
DEPTH_H = 720
COLOR_W = 1920
COLOR_H = 1080
config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.rgb8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        height, width = color_image.shape[:2]
        expected = 300
        aspect = width / height
        resized_image = cv2.resize(color_image, (round(expected * aspect), expected))
        crop_start = round(expected * (aspect - 1) / 2)
        crop_img = resized_image[0:expected, crop_start:crop_start + expected]

        inScaleFactor = 0.007843
        meanVal = 127.53
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

        blob = cv2.dnn.blobFromImage(crop_img, inScaleFactor, (expected, expected), meanVal, False)
        net.setInput(blob, "data")
        detections = net.forward("detection_out")

        min_confidence = 0.2

        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > min_confidence:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                # print(startX, startY, endX, endY)

                # draw the prediction and labels only if it's a person
                if CLASSES[idx] == "person":
                    label = "{}: {:.2f}%".format(CLASSES[idx],
                                                 confidence * 100)

                    # get a subimage to make into a gaussian blur. Make it a little bigger than
                    # min_blur_height = 0
                    # min_blur_width = 0
                    # if endY-startY < 10:
                    #     min_blur_height = 10
                    # if endX-startX < 10:
                    #     min_blur_width = 10
                    #
                    # adjusted_startY = int(startY-min_blur_height/2)
                    # adjusted_endY = int(endY+min_blur_height/2)
                    # adjusted_startX = int(startX-min_blur_width/2)
                    # adjusted_endX = int(endX+min_blur_width/2)
                    #
                    # if adjusted_startY < 0: adjusted_startY = 0
                    # if adjusted_endY > height: adjusted_endY = height-1
                    # if adjusted_startX < 0: adjusted_startX = 0
                    # if adjusted_endX > height: adjusted_endX = width-1
                    #
                    # subimg = color_image[
                    #              adjusted_startY: adjusted_endY,
                    #              adjusted_startX: adjusted_endX
                    #          ]
                    # print(adjusted_startY,adjusted_endY,adjusted_startX,adjusted_endX)
                    # blurred_subimg = cv2.GaussianBlur(subimg, (51, 51), cv2.BORDER_DEFAULT)
                    # print(blurred_subimg.shape)
                    #
                    # centerY = adjusted_endY - adjusted_startY
                    # centerX = adjusted_endX - adjusted_startX
                    # center = (centerX, centerY)
                    # print(center)
                    #
                    # # https://stackoverflow.com/questions/24195138/gaussian-blurring-with-opencv-only-blurring-a-subregion-of-an-image
                    #
                    # # replace the image
                    # # https://stackoverflow.com/questions/53929748/using-opencv-and-python-to-replace-a-segmented-part-of-an-image-with-other-image
                    #
                    # src_mask = np.zeros(blurred_subimg.shape, blurred_subimg.dtype)
                    # poly = np.array([[4, 80], [30, 54], [151, 63], [254, 37], [298, 90], [272, 134], [43, 122]],
                    #                 np.int32)
                    # cv2.fillPoly(src_mask, [poly], (255, 255, 255))

                    # color_image = cv2.seamlessClone(blurred_subimg, color_image, src_mask, center, cv2.NORMAL_CLONE)
                    # cv2.imshow("RealSense Smoothing", np.hstack((color_image, blurred_img)))

                    subimg = color_image[
                                     startY: endY,
                                     startX: endX
                                 ]

                    blurred_subimg = cv2.GaussianBlur(subimg,(99,99),0)

                    color_image[startY: endY, startX: endX] = blurred_subimg



                    # cv2.imshow("Blurred Image",color_image)

                    # cv2.rectangle(crop_img, (startX, startY), (endX, endY),
                    #               COLORS[idx], 2)
                    # cv2.rectangle(color_image, (startX, startY), (endX, endY),
                    #               COLORS[idx], 2)
                    # y = startY - 15 if startY - 15 > 15 else startY + 15
                    # cv2.putText(crop_img, label, (startX, y),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)



            # add depth to the damn thing
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            #
            # # put the bounding box on the depth image, get the depth, and add to the box on colored.
            #
            #
            # scale = height / expected
            # xmin_depth = int((xmin * expected + crop_start) * scale)
            # ymin_depth = int((ymin * expected) * scale)
            # xmax_depth = int((xmax * expected + crop_start) * scale)
            # ymax_depth = int((ymax * expected) * scale)
            # cv2.rectangle(colorized_depth, (xmin_depth, ymin_depth),
            #               (xmax_depth, ymax_depth), (255, 255, 255), 2)
            # # plt.imshow(colorized_depth)
            #
            # depth = np.asanyarray(aligned_depth_frame.get_data())
            # # Crop depth data:
            # depth = depth[xmin_depth:xmax_depth, ymin_depth:ymax_depth].astype(float)
            #
            # # Get data scale from the device and convert to meters
            # depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            # depth = depth * depth_scale
            # dist, _, _, _ = cv2.mean(depth)
            # print("Detected a {0} {1:.3} meters away.".format(className, dist))

            # Stack both images horizontally
            # images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            cv2.destroyAllWindows()
            pipeline.stop()
            break

finally:

    # Stop streaming
    pipeline.stop()