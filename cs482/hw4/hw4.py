#!/usr/bin/python

import numpy as np
import cv2
from types import *
import math
from os import listdir, path
from Queue import Queue
from threading import Thread
from random import randint

def findKeypoints(image, draw=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect keypoints in the image
    detector = cv2.FeatureDetector_create("SIFT")
    kps = detector.detect(gray)

    extractor = cv2.DescriptorExtractor_create("SIFT")
    (kps, features) = extractor.compute(gray, kps)
    
    featureImg = None
    if draw:
        featureImg = image.copy()
        for kp in kps:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            cv2.circle(featureImg, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size/2), color, 2)
            angle = math.radians(kp.angle)
            length = kp.size + 10
            cv2.line(featureImg, (int(kp.pt[0]), int(kp.pt[1])), 
                (int(kp.pt[0] + length * math.cos(angle)), int(kp.pt[1] + length * math.sin(angle))), color, 1)

    return (kps, features, featureImg)

def findMatches(image1, features1, kps1, image2, features2, kps2, ratio=.75, draw=True):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    bestMatches = matcher.knnMatch(features1, features2, 2)

    if draw:
        (h1, w1) = image1.shape[:2]
        (h2, w2) = image2.shape[:2]
        matchImg = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
        matchImg[0:h1, 0:w1] = image1
        matchImg[0:h2, w2:] = image2
    else:
        matchImg = None

    matches = []
    for match in bestMatches:
        if len(match) == 2 and match[0].distance < match[1].distance * ratio:
            trainIdx = match[0].trainIdx
            queryIdx = match[0].queryIdx
            matches.append((trainIdx, queryIdx))
            if draw:
                color = (randint(0, 255), randint(0, 255), randint(0, 255))
                pt1 = (int(kps1[queryIdx].pt[0]), int(kps1[queryIdx].pt[1]))
                pt2 = (int(kps2[trainIdx].pt[0]) + w1, int(kps2[trainIdx].pt[1]))
                cv2.line(matchImg, pt1, pt2, color, 1)
    return (matches, matchImg)

def findAffineAndHomography(matches, kps1, kps2, reProjThresh=4.0):
    if len(matches) > 4:
        pts1 = np.float32([kps1[i].pt for (_,i) in matches])
        pts2 = np.float32([kps2[i].pt for (i,_) in matches])

        (H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC, reProjThresh)
        affine = cv2.getAffineTransform(pts1[0:3], pts2[0:3])

        return (affine, H, status)

    return None

def stitch(image1, image2, H):
    result = cv2.warpPerspective(image1, H, (image1.shape[1] + image2.shape[1], image1.shape[0] + image2.shape[0]))
    result[0:image2.shape[0], 0:image2.shape[1]] = image2
    return result

if __name__ == "__main__":
    print("Loading Images")
    image1 = cv2.imread("./2010_06_10/IMG_1214.JPG")
    image2 = cv2.imread("./2010_06_10/IMG_1215.JPG")

    print("Finding Features and Keypoints")
    (kps1, features1, featureImage1) = findKeypoints(image1)
    (kps2, features2, featureImage2) = findKeypoints(image2)

    cv2.imwrite("./output/featureImage1.jpg", featureImage1)
    cv2.imwrite("./output/featureImage2.jpg", featureImage2)

    print("Finding Matches")
    (matches, matchImage) = findMatches(featureImage1, features1, kps1, featureImage2, features2, kps2, .5)

    cv2.imwrite("./output/matchImage.jpg", matchImage)

    print("Finding Affine and Homography")
    (affine, H, status) = findAffineAndHomography(matches, kps1, kps2)

    print("Affine:")
    print(affine)

    print("Homography:")
    print(H)

    print("Stitching image")
    stitchImage = stitch(image1, image2, H)

    cv2.imwrite("./output/stitchImage.jpg", stitchImage)

    print("Done")

