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

def findAllMatches(inputDir, ratio=.75):
    detector = cv2.FeatureDetector_create("SIFT")

    extractor = cv2.DescriptorExtractor_create("SIFT")

    filenames = listdir(inputDir)
    images = []
    kps = []
    features = []

    for filename in filenames:
        image = cv2.imread(inputDir + filename)
        images.append(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        (kpsTemp, featuresTemp) = extractor.compute(gray, 
                detector.detect(gray))
        kps.append(kpsTemp)
        features.append(featuresTemp)

    matcher = cv2.DescriptorMatcher_create("BruteForce")

    size = len(images)
    matrixImg = np.zeros((size, size, 1), np.float)
    for i in range(size):
        for j in range(size):
            matches = matcher.knnMatch(features[i], features[j], 2)
            bestMatches = []
            for match in matches:
                if len(match) == 2 and match[0].distance < match[1].distance * ratio:
                    trainIdx = match[0].trainIdx
                    queryIdx = match[0].queryIdx
                    bestMatches.append((trainIdx, queryIdx))
            matrixImg[i][j] = float(len(bestMatches))/(len(features[i]) + len(features[j]))

    return matrixImg

class Worker(Thread):
    '''Worker thread used to proccess mutilple images at once, 
    loads the image, proccess it, and then saves the output'''
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        while True:
            (results, name, function, arguments) = self.tasks.get()
            try:
                ret = function(*arguments)
                if results is not None:
                    results[name] = ret
            except Exception as e:
                print("Exception on:" + function.__name__ + " with arguments: " + str(arguments) + str(e))
            finally:
                self.tasks.task_done()

class ThreadPool:
    '''Thread pool which uses a queue to process incoming images'''
    def __init__(self, numberOfThreads):
        self.tasks = Queue(numberOfThreads)
        for _ in range(numberOfThreads):
            Worker(self.tasks)

    def addTask(self, function, arguments, results=None, name=None):
        self.tasks.put((results, name, function, arguments))

    def map(self, function, argumentSet, results=None, names=None):
        assert results == None or len(argumentSet) == len(names), "ArgumentSet and Names are not eqaul"
        for i in range(len(argumentSet)):
            if names is not None:
                name = names[i]
            else:
                name = None
            self.tasks.put((results, name, function, argumentSet[i]))

    def wait_finished(self):
        self.tasks.join()

if __name__ == "__main__":
    inputDir = "./lab3pics/"
    outputDir = "./output/"
    pool = ThreadPool(9)

    results = {}
    for filename in listdir(inputDir):
        image = cv2.imread(inputDir + filename)
        pool.addTask(findKeypoints, (image, True), results, filename)

    pool.wait_finished()

    for key, result in results.iteritems():
        cv2.imwrite(outputDir + "/features/" + key, result[2])
    
    goodMatch = findMatches(
            results["ST2MainHall4001.jpg"][2], results["ST2MainHall4001.jpg"][1], results["ST2MainHall4001.jpg"][0], 
            results["ST2MainHall4015.jpg"][2], results["ST2MainHall4015.jpg"][1], results["ST2MainHall4015.jpg"][0], .75, True)
    badMatch = findMatches(
            results["ST2MainHall4001.jpg"][2], results["ST2MainHall4001.jpg"][1], results["ST2MainHall4001.jpg"][0], 
            results["ST2MainHall4035.jpg"][2], results["ST2MainHall4035.jpg"][1], results["ST2MainHall4035.jpg"][0], .75, True)
    cv2.imwrite(outputDir + "/matches/goodMatch.jpg", goodMatch[1])
    cv2.imwrite(outputDir + "/matches/badMatch.jpg", badMatch[1])


    output = findAllMatches(inputDir)
    im = np.array(output * 255, dtype = np.uint8)
    threshold = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    cv2.imwrite(outputDir + "match-matrix-gradient.jpg", im)
    cv2.imwrite(outputDir + "match-matrix-min-max.jpg", threshold)
    np.savetxt(outputDir + "match-matrix.txt", output)
