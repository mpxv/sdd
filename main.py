import numpy as np
import cv2
import constants
import math


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
frame=cv2.imread('dataset/test0.jpeg')

found, w = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)

found = np.sort(found, axis=0)

draw_detections(frame,found)
cv2.imshow('feed',frame)

output = []

for p1,p2 in zip(found[:-1],found[1:]):
    output.append([p2[0]-p1[0],p1[3]]) # dist, height

i = 1
for pair in output:
    # pair[0] = distance between the person and the next person in px
    # pair[1] = person's height in px
    distance = (constants.AVG_HEIGHT / pair[1]) * pair[0]
    if distance < 183: # if less than 183 cm or 6 feet apart
        print("Pair " + str(i) + ": Unsafe! Distance: " + str(round(distance)) + "cm")
    else:
        print("Pair " + str(i) + ": Safe! Distance: " + str(round(distance)) + "cm")
    i += 1

while True:
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break
cv2.destroyAllWindows()



