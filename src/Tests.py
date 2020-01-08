import cv2 as cv
import numpy as np

FILE_NAME = "../images/page.jpg"
RESIZE_RATIO = 0.65


# Tracks over all words
def onTrack(data):
    # Copy image for display purposes
    temp = img.copy()

    # Draw center point of word
    M = cv.moments(cont[data])
    cx = int(M['m10'] / (M['m00'] + 1E-5))
    cy = int(M['m01'] / (M['m00'] + 1E-5))
    cv.circle(temp, (cx, cy), 2, (0, 0, 255), 2)

    # Draw bounding box around word, based on far left and right
    # and the top most and bottom most pixel
    x, y, w, h = cv.boundingRect(cont[data])
    cv.rectangle(temp, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv.imshow("Image", temp)
    cv.imwrite("one_words.png", temp)


cv.namedWindow("Image")
# Read in image and resize to fit screen
img = cv.imread(FILE_NAME, cv.IMREAD_COLOR)
img = cv.resize(img, (0, 0), fx=RESIZE_RATIO, fy=RESIZE_RATIO)

# Use Canny to get edges of characters
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.bilateralFilter(gray, 11, 75, 75)
edge = cv.Canny(gray, 50, 150)

# Show image pre any morphological methods
cv.imshow("Pre-Edge", edge)

# Dilate to connect letters
kernel = np.ones((7, 7), np.uint8)
edge = cv.morphologyEx(edge, cv.MORPH_DILATE, kernel)

# Get original contours, this will include holes in letters like "O" or "P"
cont, heir = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Fill in any holes using convex
for i in range(0, len(cont)):
    cv.fillConvexPoly(edge, cont[i], 255)

# Use erode to disconnect lose connections
edge = cv.morphologyEx(edge, cv.MORPH_ERODE, kernel)

# Get the new contours
cont, heir = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Show image so trackbar shows correctly and set up trackbar
cv.imshow("Image", img)
cv.createTrackbar("Word", "Image", 0, len(cont)-1, onTrack)

# Draw the bounding boxes for all the words, and the center points
img2 = img.copy()
for i in range(0, len(cont)):
    M = cv.moments(cont[i])
    cx = int(M['m10'] / (M['m00'] + 1E-5))
    cy = int(M['m01'] / (M['m00'] + 1E-5))
    cv.circle(img2, (cx, cy), 2, (0, 0, 255), 2)

    x, y, w, h = cv.boundingRect(cont[i])
    cv.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 1)

onTrack(0)
cv.imshow("Words", img2)
cv.imwrite("all_words.png", img2)
cv.imshow("Edge", edge)
cv.waitKey(0)
cv.destroyAllWindows()