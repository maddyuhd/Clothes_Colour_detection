# import all the necessary modules to work with
from __future__ import division
import argparse
import cv2 
import numpy as np
import time as t
from pprint import pprint
from sklearn.cluster import KMeans



ap = argparse.ArgumentParser()

number_of_outputs = 3 # number of output rgb values you want to display

ap.add_argument("-i", "--image", required=True,
                help="path to input image")

ap.add_argument('-d',"--debug",action="store_true", 
    help="to debug the program",default=False)

ap.add_argument('-t',"--time",action="store_true", 
    help="to check the time taken to process",default=False)

args = vars(ap.parse_args())

def show(im):
    msg = 'press any key to continue'
    cv2.namedWindow(msg, cv2.WINDOW_NORMAL)
    cv2.imshow(msg, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hig_val(d, v):
    loc = []

    for c in range(0, v):
        m = max(d, key=d.get)
        loc.append(m)
        del d[m]
    return loc

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
 
    # return the histogram
    return hist

def process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters = number_of_outputs)
    clt.fit(image)
    hist = centroid_histogram(clt)
    return hist,clt


result = {}

start = t.time()
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 

img = cv2.imread(args["image"])

orig = img.copy()

r = 300.0 / img.shape[1]
dim = (300, int(img.shape[0] * r))

# perform the actual resizing of the image and show it
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

new_h = resized.shape[0]
new_w = resized.shape[1]

(rects, weights) = hog.detectMultiScale(resized, winStride=(4, 4),
    padding=(8, 8), scale=1.05)

if (len(rects) !=0):
    # draw the original bounding boxes
    # for (x, y, w, h) in rects:
    #     cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    total_area = new_h * new_w

    # draw the final bounding boxes
    for (x,y,w,h) in pick:

        # cv2.rectangle(resized, (x, y), (w, h), (0, 255, 0), 2)
        roi_color = resized[y:h, x:w]

        #checking the detected area
        if (args["debug"]):
            print "area recog: ",((roi_color.shape[0]*roi_color.shape[1])/total_area*100)
        if ((roi_color.shape[0]*roi_color.shape[1])/total_area*100) > 15:

            # orig = np.concatenate((resized, resized1), axis=0)
            h, clt = process(roi_color)
            
            for idx, score in enumerate(h.tolist()):
                result[score] = clt.cluster_centers_.tolist()[idx]

            if (args["debug"]):
                print "                                          "
                print "----------- Highest likelihood -----------"
                print "------------------------------------------"
                print "score :",h.tolist()
                print "                                          "
                print "------------- Raw Pixel Data -------------"
                print "------------------------------------------"
                print "rbg val :", clt.cluster_centers_.tolist()
                show(roi_color)
            
            break

        else:

            offset = 40

            roi_color = resized[int((new_h/2))-offset:int((new_h/2))+offset, int((new_w/2)) - offset : int((new_w/2)) + offset]

            h, clt = process(roi_color)
            for idx, score in enumerate(h.tolist()):
                result[score] = clt.cluster_centers_.tolist()[idx]

            if (args["debug"]):
                print "                                          "
                print "----------- Highest likelihood -----------"
                print "------------------------------------------"
                print "score :",h.tolist()
                print "                                          "
                print "------------- Raw Pixel Data -------------"
                print "------------------------------------------"
                print "rbg val :", clt.cluster_centers_.tolist()
                show(roi_color)
            break


else:

    face_model = cv2.CascadeClassifier('dev_model.xml')

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # # Detects faces of different sizes in the input image
    faces = face_model.detectMultiScale(gray, 1.3, 5)

    if len(faces)!=0 :

        hig = gray.shape[0]

        for (x,y,w,h) in faces:

            # if (args["debug"]):
            #     # To draw a rectangle in a face 
            #     cv2.rectangle(resized,(x,y),(x+w,y+h),(255,255,0),2) 

            roi_color = resized[y+h : hig , x:x+w]

            h, clt = process(roi_color)
            
            for idx, score in enumerate(h.tolist()):
                result[score] = clt.cluster_centers_.tolist()[idx]

            if (args["debug"]):
                print "                                          "
                print "----------- Highest likelihood -----------"
                print "------------------------------------------"
                print "score :",h.tolist()
                print "                                          "
                print "------------- Raw Pixel Data -------------"
                print "------------------------------------------"
                print "rbg val :", clt.cluster_centers_.tolist()
                show(roi_color)

    else :

        offset = 40

        roi_color = resized[int((new_h/2))-offset:int((new_h/2))+offset, int((new_w/2)) - offset : int((new_w/2)) + offset]

        h, clt = process(roi_color)
        for idx, score in enumerate(h.tolist()):
            result[score] = clt.cluster_centers_.tolist()[idx]

        if (args["debug"]):
            print "                                          "
            print "----------- Highest likelihood -----------"
            print "------------------------------------------"
            print "score :",h.tolist()
            print "                                          "
            print "------------- Raw Pixel Data -------------"
            print "------------------------------------------"
            print "rbg val :", clt.cluster_centers_.tolist()
            show(roi_color)

#convert float to int
for i in result:
    result[i]= [int(j) for j in result[i]]

#sort the data 
final_idx = sorted(result, key=result.get, reverse=True)[:number_of_outputs]

if (args["debug"]):
    print "                                          "
    print " Final result : "
    print "[ R , G , B ] <-- Highest likelihood"

for indx in final_idx:
    print result[indx] 

if (args["debug"]):
    show(resized)

if (args["time"]):
    print "Time taken : {} sec".format(t.time()-start)