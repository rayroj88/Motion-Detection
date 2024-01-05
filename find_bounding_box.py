import cv2 as cv
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from parse_frame_name import parse_frame_name
from make_frame_name import make_frame_name
from read_gray import read_gray

def find_bounding_box(frame_path):
    MIN_FRAME = 0
    MAX_FRAME = 124

    # Parse frame name
    #Turns a single frame filepath into a directory and frame number
    dir_name, frame_number = parse_frame_name(frame_path)
    
    #Turns frame numbers into filenames
    frame1 = make_frame_name(dir_name, frame_number)
    frame2 = make_frame_name(dir_name, frame_number - 10)
    frame3 = make_frame_name(dir_name, frame_number + 10)
    
    #Takes file_path as input and reads in as a gray image returns a frame number
    frame1 = read_gray(frame1)
    frame2 = read_gray(frame2)
    frame3 = read_gray(frame3)
    
    #Output image for troubleshooting
    #img = cv.imread(frame1)
    #while True:
    #    cv.imshow("frame1", img)
    #    cv.waitKey(0)
    #    exit()
    #cv.destroyAllWindows()
    
    
    #Takes frame numbers as input
    diff1 = cv.absdiff(frame1, frame2)
    diff2 = cv.absdiff(frame1, frame3)
    
    motion = cv.min(diff1, diff2)

    thresh, binary_img = cv.threshold(motion, thresh = 10, maxval = 255, type = cv.THRESH_BINARY)
    
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(binary_img, connectivity=4)
    
    max_label, max_size = max([(i, stats[i, cv.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
    
    binary_img[output != max_label] = 0
    
    left = int(stats[max_label, cv.CC_STAT_LEFT])
    top = int(stats[max_label, cv.CC_STAT_TOP])
    width = int(stats[max_label, cv.CC_STAT_WIDTH])
    height = int(stats[max_label, cv.CC_STAT_HEIGHT])
    
    right_column = left + width
    bottom_row = top + height
    left_column = left
    top_row = top

    frame = cv.cvtColor(binary_img, cv.COLOR_BGR2RGB)
    
    cv.rectangle(frame, (left, top), (left+width, top+height), (0,255,255), 5)
    
    
    # Save the motion image with the bounding box as a file
    cv.imwrite('output/detection_frame'+str(frame_number)+'.jpg', frame)
    
    return (top_row, bottom_row, left_column, right_column)
