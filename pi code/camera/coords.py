import os
import sys
import argparse
import glob
import time

import cv2 as cv
import numpy as np
from ultralytics import YOLO

# define user input arguments

parser = argparse.ArgumentParser()


parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected cards (example: "0.4")',
                    default=0.5)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()


# parse user inputs
model_path = "runs/detect/train/weights/best.pt"
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# ARUCO MARKERS 
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
parameters = cv.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX


detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

dist_13 = 75.0

dist_49 = 75.0


# check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# load model 
model = YOLO(model_path, task='detect')
labels = model.names

# parse resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# set up recording
if record:
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv.VideoWriter(record_name, cv.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

cap = cv.VideoCapture(0)

# set camera or video resolution if specified by user
if user_res:
    cap.set(3, resW)
    cap.set(4, resH)

# set bounding box colours
bbox_colours = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# begin inference loop
while True:


    t_start = time.perf_counter()


    ret, frame = cap.read()
    if (frame is None) or (not ret):
        print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
        break


    # resize frame
    if resize == True:
        frame = cv.resize(frame,(resW,resH))

    # run inference on frame
    results = model(frame, verbose=False)

    # extract results
    detections = results[0].boxes

    card_count = 0
    card_centres = {}

    # go through each detection and get bbox coords, confidence and class
    for i in range(len(detections)):

        # get bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze() # convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # extract individual coordinates and convert to int

        # get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # get bounding box confidence
        conf = detections[i].conf.item()

        centre = (int((xmax + xmin) / 2), int((ymax + ymin) / 2))

        if conf > min_thresh:

            colour = bbox_colours[classidx % 10]
            cv.rectangle(frame, (xmin,ymin), (xmax,ymax), colour, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1) # get font size
            label_ymin = max(ymin, labelSize[1] + 10) # buffer
            cv.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), colour, cv.FILLED) # draw white box to put label text in
            cv.putText(frame, label, (xmin, label_ymin-7), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # draw label text

            radius = max(5, int(min(xmax - xmin, ymax - ymin) / 4))
            cv.circle(frame, centre, radius, colour, -1)

            card_centres[classname] = centre

            card_count = card_count + 1


    # ARUCO MARKERS:
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(grey)
    
    if ids is not None:
        cv.aruco.drawDetectedMarkers(frame, corners, ids)

        marker_centres = {}
        
        for i, corner in enumerate(corners):

            # reshape array (4 rows, 2 columns)
            pts = corner.reshape((4, 2))

            centre_x = int(pts[:, 0].mean())
            centre_y = int(pts[:, 1].mean())
            
            marker_id = int(ids[i][0])

            # add marker centre coords to dict
            marker_centres[marker_id] = (centre_x, centre_y)

            # draw circle at centre of aruco marker
            cv.circle(frame, (centre_x, centre_y), 15, (0, 0, 255), -1)

        if 13 in marker_centres and 49 in marker_centres:
            # draw line connecting centres of 13 and 49
            cv.line(frame, marker_centres[13], marker_centres[49], (255, 255, 0), 3)

            # midpoint of markers is the centre of the base
            midpoint = ((marker_centres[13][0] + marker_centres[49][0]) // 2, (marker_centres[13][1] + marker_centres[49][1]) // 2)

            dy = marker_centres[13][1] - marker_centres[49][1]
            dx = marker_centres[13][0] - marker_centres[49][0]

            # the difference in rotation between the camera and the base
            theta = -(np.arctan2(dy, dx) + np.pi)


            if theta != 0:
                transform = np.array([
                    [np.cos(theta), -np.sin(theta)], 
                    [np.sin(theta), np.cos(theta)]
                ])

            else:
                transform = np.array([
                    [1, 0], # THE IDENTITY MATRIX!!!!
                    [0, 1]
                ])    

            

            for card in card_centres:
                cv.line(frame, midpoint, card_centres[card], (0, 0, 255), 3)

                # difference between position of card and position of base
                diff_coords = np.array([
                    card_centres[card][0] - midpoint[0],
                    card_centres[card][1] - midpoint[1]
                ])
                
                # with the base set as the "origin", rotate the card by theta
                rotated = np.dot(transform, diff_coords)

                # 540mm = 720 pixels
                
                # coords of the card in the coordinate frame of the base
                robot_coords = (round(float(rotated[0]) * (540/720), 2), round(-float(rotated[1]) * (540/720), 2))


                print(f"{card} relative coords: {robot_coords}")

                # creates a point to show the rotation
                rotation_point = (int(rotated[0] + midpoint[0]), int(rotated[1] + midpoint[1]))
                cv.circle(frame, rotation_point, 5, (0, 255, 0), -1)
                            

    # calculate and draw framerate
    cv.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # draw framerate
    cv.putText(frame, f'Resolution: {frame.shape[1]}x{frame.shape[0]}', (10,40), cv.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # draw resolution
    
    # display detection results
    cv.putText(frame, f'Number of cards: {card_count}', (10,60), cv.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # draw total number of detected cards
    cv.imshow('YOLO detection results',frame) # display image
    if record: recorder.write(frame)

    # wwit 5ms 
    key = cv.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): 
        break
    elif key == ord('s') or key == ord('S'): # press 's' to pause inference
        cv.waitKey()
    elif key == ord('p') or key == ord('P'): # press 'p' to save a picture of results on this frame
        cv.imwrite('capture.png',frame)
    
    # calculate fps for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # append fps result to frame_rate_buffer (for finding average fps over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # mean fps
    avg_frame_rate = np.mean(frame_rate_buffer)


print(f'Average pipeline FPS: {avg_frame_rate:.2f}')

cap.release()
cv.destroyAllWindows()
