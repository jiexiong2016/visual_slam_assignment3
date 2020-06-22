import numpy as np
import cv2

from collections import defaultdict
from numbers import Number

from threading import Thread, Lock 
from queue import Queue

class Params(object):
    def __init__(self):
        self.feature_detector = cv2.GFTTDetector_create(
            maxCorners=600, minDistance=15.0, 
            qualityLevel=0.001, useHarrisDetector=False)
        self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
            bytes=32, use_orientation=False)
        self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.matching_distance = 30
        self.matching_distance_ratio = 0.8

        self.depth_near = 0.1
        self.depth_far = 10
        
        self.lc_min_inbetween_keyframes = 2   # frames
        self.lc_max_inbetween_distance = 3  # meters
        self.lc_embedding_distance = 30
        self.lc_inliers_threshold = 13
        self.lc_inliers_ratio = 0.3

        self.view_camera_width = 0.05
        self.view_viewpoint_x = 0
        self.view_viewpoint_y = -1
        self.view_viewpoint_z = -10
        self.view_viewpoint_f = 2000
        self.view_image_width = 320
        self.view_image_height = 240

        self.view_point_cloud = False

class ImageFeature(object):
    def __init__(self, image, params):
        self.image = image 
        self.height, self.width = image.shape[:2]

        self.keypoints = []      # list of cv2.KeyPoint
        self.descriptors = []    # numpy.ndarray

        self.detector = params.feature_detector
        self.extractor = params.descriptor_extractor
        self.matcher = params.descriptor_matcher
        self.matching_distance = params.matching_distance

        self._lock = Lock()

    def extract(self):
        self.keypoints = self.detector.detect(self.image)
        self.keypoints, self.descriptors = self.extractor.compute(
            self.image, self.keypoints)

    def direct_match(self, desps1, desps2, matching_distance=30, ratio=0.7):
        matches = dict()
        distances = defaultdict(lambda: float('inf'))

        for (m, n) in self.matcher.knnMatch(np.array(desps1), np.array(desps2), k=2):
            if m.distance < min(
                matching_distance, n.distance * ratio, distances[m.trainIdx]):
                matches[m.trainIdx] = m.queryIdx
                distances[m.trainIdx] = m.distance

        return [(i, j) for j, i in matches.items()]

    def get_color(self, pt):
        x = int(np.clip(pt[0], 0, self.width-1))
        y = int(np.clip(pt[1], 0, self.height-1))
        color = self.image[y, x]
        if isinstance(color, Number):
            color = np.array([color, color, color])
        return color[::-1] / 255.