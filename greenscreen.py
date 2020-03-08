import numpy as np                        # fundamental package for scientific computing
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import cv2
import numba as nb

print("Environment Ready")

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

DEPTH_W = 848
DEPTH_H = 480
COLOR_W = 848
COLOR_H = 480
config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, 30)

# Set green screen background.
GREENSCREEN_DEPTH = 3  # in meters
bg_img_r = np.full((COLOR_H,COLOR_W),0)
bg_img_g = np.full((COLOR_H,COLOR_W),255)
bg_img_b = np.full((COLOR_H,COLOR_W),0)
bg_img = np.dstack((bg_img_r, bg_img_g, bg_img_b))
print(f"Background image shape: {bg_img.shape}")

# or Select an image here


@nb.njit(parallel=True)
def assemble_greenscreen_img(rs_img, bg_img, depth_array, green_screen_threshold=GREENSCREEN_DEPTH):
    output_img = np.empty(rs_img.shape).astype(np.uint8)
    for i in nb.prange(rs_img.shape[0]):
        for j in nb.prange(rs_img.shape[1]):
            if depth_array[i, j] < green_screen_threshold:
                output_img[i, j] = rs_img[i, j].astype(np.uint8)
            else:
                output_img[i, j] = bg_img[i, j].astype(np.uint8)
    return output_img

# Start streaming
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()


        # Convert images to numpy arrays
        # depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        align = rs.align(rs.stream.color)
        frames = align.process(frames)
        aligned_depth_frame = frames.get_depth_frame()
        depth = np.asanyarray(aligned_depth_frame.get_data())

        if not aligned_depth_frame or not color_frame:
            continue

        depth = depth * depth_scale  # depth array in meters
        dist, _, _, _ = cv2.mean(depth)

        merged_img = assemble_greenscreen_img(color_image, bg_img, depth)


        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', merged_img)

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            cv2.destroyAllWindows()
            pipeline.stop()
            break

finally:

    # Stop streaming
    cv2.destroyAllWindows()
    pipeline.stop()
