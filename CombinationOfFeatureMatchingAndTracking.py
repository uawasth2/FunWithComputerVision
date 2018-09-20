import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import sys
 
img1 = cv2.imread('cornercast.png', 0)

# detector of similarities
orb = cv2.ORB_create()

# this will find the region that contains the most features of the image we want it to detect
def most_match_finder(img, kp, matches):
    # best x and y coords in image
    best_x = 0
    best_y = 0
    # the maximum features in the region
    max_match = 0
    # the size of the region
    range_ = 100

    # resorts to make sure that matches came in sorted
    matches = sorted(matches, key = lambda x:x.distance)

    # y and x are switched because cv reads images differently than numpy allows you to view the images
    for y in range(0, len(img), range_):
        for x in range(0, len(img[y]), range_):
            # goes through all matches found and sees how many are in your current region
            curr_match = 0
            for m in matches[:10]:
                loc1, loc2 = kp[m.trainIdx].pt
                cv2.circle(img, ( int(math.floor(loc1)), int(math.floor(loc2)) ), 20, 0, -1)
                if loc1 >= x and loc1 <= x + range_ and loc2 >= y and loc2 <= y + range_:
                    curr_match += 1
            # if more matches found replace the best match with the current one
            if curr_match > max_match:
                best_x = x
                best_y = y
                max_match = curr_match
    cv2.rectangle(img, (best_x, best_y), (best_x + range_, best_y + range_), 0, 5) 
    # returns x, y, and side size of range
    return best_x, best_y, range_


if __name__ == '__main__' :
 
    # Set up tracker.
    # Instead of MIL, you can also use
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]
 
    if True:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
 
    # Read video
    video = cv2.VideoCapture(0)
    
    # Exit if video not opened.
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print ('Cannot read video file')
        sys.exit()

    # takes first frame and finds matching points and then draws box needed for tracking
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # keypoint and descriptors of each image
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(frame, None)

    # find keypoints and descriptors now
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    
    # find and sort matches based on distance/accuracy/confidence
    matches = bf.match(des1, des2)
    # most likely to leas likely to match
    matches = sorted(matches, key = lambda x:x.distance)

        
    print(len(kp1))
    
    in_x, in_y, size = most_match_finder(frame, kp2, matches)
     
    # Define an initial bounding box
    bbox = (in_x, in_y, size, size)
 
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    count = 0
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

            # will count for how long an error exists
            count = 0
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            count += 1

            # checks to make sure that interference was not temporary
            if (count > 100):
                tracker.clear()
                tracker = cv2.TrackerKCF_create()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # keypoint and descriptors of each image
                kp1, des1 = orb.detectAndCompute(img1, None)
                kp2, des2 = orb.detectAndCompute(frame, None)

                # find keypoints and descriptors now
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

                # find and sort matches based on distance/accuracy/confidence
                matches = bf.match(des1, des2)
                # most likely to leas likely to match
                matches = sorted(matches, key = lambda x:x.distance)

                in_x, in_y, size = most_match_finder(frame, kp2, matches)

                # redefine an initial bounding box
                bbox = (in_x, in_y, size, size)
                ok = tracker.init(frame, bbox)
            
 
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
