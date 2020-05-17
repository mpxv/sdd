# USAGE
# python distance_to_camera.py

# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2

def find_marker(image, num_people = 3):
	# convert the image to grayscale, blur it, and detect edges
	grey_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	grey_scale = cv2.GaussianBlur(grey_scale, (5, 5), 0)
	edged = cv2.Canny(grey_scale, 70, 125)

	# find the contours in the edged image and keep the largest ones
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

	c = cnts[:num_people]
	rectangles = []
	for c in c:
		print(cv2.minAreaRect(c))
		rectangles.append(cv2.minAreaRect(c))
	print(rectangles)
	# compute the areas of all the triangles of people
	return rectangles


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

draw_detections(frame, found)
cv2.imshow('feed', frame)


def distance_from_edge(actual_width, focal_length, pixel_width):
	return (actual_width * focal_length / pixel_width)

KNOWN_DISTANCE = 144
KNOWN_WIDTH = 12.0


image = cv2.imread("/Users/stanley/Hackathons/HTHS Hackathon/sdd/dataset/test0.jpeg")

focalLengths = []
inches = []

for i in range(3):
	focalLength = (marker[i][1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
	focalLengths.append(focalLength)
	inches.append(distance_from_edge(KNOWN_WIDTH, focalLengths[i], marker[i][1][0]))
	box = cv2.cv.BoxPoints(marker[i]) if imutils.is_cv2() else cv2.boxPoints(marker[i])
	box = np.int0(box)
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)




cv2.imshow("image", image)
cv2.waitKey(0)
