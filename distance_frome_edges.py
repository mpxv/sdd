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

def distance_from_edge(actual_width, focal_length, pixel_width):
	return (actual_width * focal_length / pixel_width)

KNOWN_DISTANCE = 144
KNOWN_WIDTH = 12.0


image = cv2.imread("/Users/stanley/Hackathons/HTHS Hackathon/sdd/dataset/test0.jpeg")
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH


marker = find_marker(image)
inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

# Draw a box around the human subject
box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
box = np.int0(box)
cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
cv2.putText(image, "%.2fft" % (inches / 12),
	(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
	2.0, (0, 255, 0), 3)




cv2.imshow("image", image)
cv2.waitKey(0)
