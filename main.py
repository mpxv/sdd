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
        
def draw_line(img, p1, p2, text, color):
    xy1=tuple(p1[:2]+p1[2:]//2)
    xy2=tuple(p2[:2]+p2[2:]//2)
    cv2.line(img, xy1, xy2, color)
    # Line thickness of 2 px 
    # breakpoint()
    cv2.putText(img, text, tuple(np.mean([xy1,xy2],axis=0,dtype=int)+np.array([-30,-10])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,255,255], 1) 

hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

frame=cv2.imread('dataset/test7.png')

found,w=hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)

found = np.sort(found, axis=0)

draw_detections(frame,found)

output = []

for p1,p2 in zip(found[:-1],found[1:]):
    output.append([p2[0]-p1[0],p1[3],p1,p2]) # dist, height

i = 1
for group in output:
    # group[0] = distance between the person and the next person in px
    # group[1] = person's height in px
    # group[2] = person 1
    # group[3] = person 2
    distance = (constants.AVG_HEIGHT / group[1]) * group[0]
    p1 = group[2]
    p2 = group[3]

    if distance < 183: # if less than 183 cm or 6 feet apart
        print("Pair " + str(i) + ": Unsafe! Distance: " + str(round(distance)) + "cm")
        # breakpoint()
        draw_line(frame, p1, p2, 'unsafe', [0,0,255])
    else:
        print("Pair " + str(i) + ": Safe! Distance: " + str(round(distance)) + "cm")
        draw_line(frame, p1, p2, 'safe', [0,255,0])
    i += 1

cv2.imshow('feedp',frame)

while True:
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break
cv2.destroyAllWindows()



