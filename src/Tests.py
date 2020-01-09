import cv2 as cv
import numpy as np
from src.Word import Word
from operator import itemgetter

FILE_NAME = "../images/page.jpg"
RESIZE_RATIO = 0.65


def getWordOrder(words, start, stop):
    print(start, stop)
    wordsInRange = list(filter(lambda w: start <= w.getCenter()[1] <= stop, words))
    sortedWords = sorted(wordsInRange, key=lambda w: w.getCenter()[0])
    return sortedWords

# Tracks over all words
def onTrack(data):
    # Copy image for display purposes
    temp = img.copy()

    # Draw center point of word
    word = words[data]
    cv.circle(temp, word.getCenter(), 2, (0, 0, 255), 2)

    # Draw bounding box around word, based on far left and right
    # and the top most and bottom most pixel
    x, y, w, h = cv.boundingRect(word.getContour())
    cv.rectangle(temp, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv.imshow("Image", temp)


cv.namedWindow("Image")
# Read in image and resize to fit screen
img = cv.imread(FILE_NAME, cv.IMREAD_COLOR)
img = cv.resize(img, (0, 0), fx=RESIZE_RATIO, fy=RESIZE_RATIO)
img2 = img.copy()

# Use Adaptive Thresholding to get individual characters
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.bilateralFilter(gray, 11, 75, 75)
edge = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 12)

# Show image pre any morphological methods
# cv.imshow("Pre-Edge", edge)

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
words = []
for i in range(0, len(cont)):
    words.append(Word(cont[i]))
    x, y, w, h = cv.boundingRect(cont[i])
    cv.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 1)

# whiteCounts helps separate the sentences
whiteCount = [0] * gray.shape[0]
for r in range(0, gray.shape[0]):
    count = 0
    for c in range(0, gray.shape[1]):
        if edge[r, c] != 0:
            count += 1
    whiteCount[r] = count

# Sort words by sentences
orderedWords = []
i = 0
while i < len(whiteCount):
    if whiteCount[i] != 0:
        startIndex = i
        while i < len(whiteCount) and whiteCount[i] != 0:
            i += 1
        orderedWords.extend(getWordOrder(words, startIndex, i))
    else:
        i += 1
words = orderedWords

# Draw lines to show order of words
prev = None
for word in words:
    if prev is not None:
        cv.line(img2, prev.getCenter(), word.getCenter(), (255, 255, 0), 2)
    prev = word

# Draw center points of words
for word in words:
    cv.circle(img2, word.getCenter(), 2, (0, 0, 255), 2)

onTrack(0)
cv.imshow("Words", img2)
# cv.imshow("Edge", edge)
cv.waitKey(0)
cv.destroyAllWindows()