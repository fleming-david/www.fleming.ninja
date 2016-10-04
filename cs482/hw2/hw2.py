#!/usr/bin/python

import numpy as np
import cv2
from types import *
import math
from os import listdir, path
from Queue import Queue
from threading import Thread
from random import randint

def findLines(orginalImg):
    '''This is the function for the homework. This function will apply
    Canny Edge Detection, Hough Transform, and Probablistic Hough Transform 
    to the passed image'''
    blurredImg = cv2.GaussianBlur(orginalImg, (7, 7), 1)
    cannyImg = cv2.Canny(blurredImg, 80, 120)

    hufflines = cv2.HoughLines(cannyImg, 1, math.pi/180, 150)
    huffImage = orginalImg.copy()
    for rho,theta in hufflines[0]:
        a, b = math.cos(theta), math.sin(theta)
        x0, y0 = a*rho, b*rho
        pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
        pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
        cv2.line(huffImage, pt1, pt2, (randint(0, 255), randint(0, 255), randint(0, 255)))

    huffPlines = cv2.HoughLinesP(cannyImg, 1, math.pi/180, 75, 100, 15)
    huffPImage = orginalImg.copy()
    for x0,y0,x1,y1 in huffPlines[0]:
        cv2.line(huffPImage, (x0, y0), (x1, y1), (randint(0, 255), randint(0, 255), randint(0, 255)), 3, 8)

    return (blurredImg, cannyImg, huffImage, huffPImage)

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
            filename = self.tasks.get()
            try:
                assert type(filename) is StringType, "filename must be a string"
                orginal = cv2.imread("./Original Images/" + filename)
                blurred, canny, huff, huffp = findLines(orginal);
                filenameOnly = path.splitext(filename)[0];
                cv2.imwrite("./output/" + filenameOnly + "_blurred.jpg", blurred)
                cv2.imwrite("./output/" + filenameOnly + "_canny.jpg", canny)
                cv2.imwrite("./output/" + filenameOnly + "_huff.jpg", huff)
                cv2.imwrite("./output/" + filenameOnly + "_huffp.jpg", huffp)
            except Exception as e:
                print("Exception on:" + filename + str(e))
            finally:
                self.tasks.task_done()

class ThreadPool:
    '''Thread pool which uses a queue to process incoming images'''
    def __init__(self, numberOfThreads):
        self.tasks = Queue(numberOfThreads)
        for _ in range(numberOfThreads):
            Worker(self.tasks)

    def map(self, filenames):
        for filename in filenames:
            self.tasks.put((filename))

    def wait_finished(self):
        self.tasks.join()

if __name__ == "__main__":
    pool = ThreadPool(9)

    pool.map(listdir("./Original Images/"))

    pool.wait_finished()

    #cv2.namedWindow("Output")
    #output = []
    #output.append((1, findLines("./Original Images/ST2MainHall4001.jpg")))
    #output.append((1, findLines("./Original Images/ST2MainHall4002.jpg")))
    ##for image in listdir("./Original Images"):
    ##    output.append((image, findLines("./Original Images/" + image)))
    #close = False
    #image = 0
    #show = 0
    #length = len(output)
    #while(not close and length > 0):
    #    cv2.imshow("Output", output[image][1][show]);
    #    key = cv2.waitKey(0) & 0xFF
    #    if key == 27 or key == 255:
    #        close = True
    #    elif key == ord('n'):
    #        image = (image + 1) % length
    #    elif key == ord('p'):
    #        image = (image - 1) % length
    #    elif key == ord('1'):
    #        show = 0
    #    elif key == ord('2'):
    #        show = 1
    #    elif key == ord('3'):
    #        show = 2
    #    elif key == ord('4'):
    #        show = 3
    #cv2.destroyAllWindows()'''
