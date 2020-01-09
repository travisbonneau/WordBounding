import cv2 as cv

class Word:
    def __init__(self, contour):
        self.contour = contour
        self.moment = cv.moments(contour)

    def getContour(self):
        return self.contour

    def getMoment(self):
        return self.moment

    def getCenter(self):
        cx = int(self.moment['m10'] / (self.moment['m00'] + 1E-5))
        cy = int(self.moment['m01'] / (self.moment['m00'] + 1E-5))
        return cx, cy

    def __str__(self):
        return "(" + str(self.getCenter()[0]) + ", " + str(self.getCenter()[1]) + ")"