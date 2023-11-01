import numpy as np
import cv2

class ball_detector:
    def __init__(self):
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 100
        # Filter by Color.
        params.filterByColor = True
        params.blobColor = 255
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 800
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.5
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.7
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.4
        # Create a detector with the parameters
        self.detector = cv2.SimpleBlobDetector_create(params)
        
    def detect(self, gray):
        keypoints = self.detector.detect(gray)
        return keypoints