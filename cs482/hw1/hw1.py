#!/usr/bin/python
import numpy
import cv2

KERNEL = (15, 15)

img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)

cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Image', img)

imgBlurred1 = cv2.GaussianBlur(img, KERNEL, 1)
imgBlurred2 = cv2.GaussianBlur(img, KERNEL, 2)
imgBlurred3 = cv2.GaussianBlur(img, KERNEL, 3)

imgDerivativesX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
imgDerivativesY = cv2.Sobel(img, cv2.CV_64F, 0, 1)

imgDerivativesX1 = cv2.Sobel(imgBlurred1, cv2.CV_64F, 1, 0)
imgDerivativesY1 = cv2.Sobel(imgBlurred1, cv2.CV_64F, 0, 1)

imgDerivativesX2 = cv2.Sobel(imgBlurred2, cv2.CV_64F, 1, 0)
imgDerivativesY2 = cv2.Sobel(imgBlurred2, cv2.CV_64F, 0, 1)

imgDerivativesX3 = cv2.Sobel(imgBlurred3, cv2.CV_64F, 1, 0)
imgDerivativesY3 = cv2.Sobel(imgBlurred3, cv2.CV_64F, 0, 1)

imgMagnitude = cv2.magnitude(imgDerivativesX, imgDerivativesY)
imgMagnitude1 = cv2.magnitude(imgDerivativesX1, imgDerivativesY1)
imgMagnitude2 = cv2.magnitude(imgDerivativesX2, imgDerivativesY2)
imgMagnitude3 = cv2.magnitude(imgDerivativesX3, imgDerivativesY3)

cv2.imwrite('./blurred/imageBlurred1.jpg', imgBlurred1)
cv2.imwrite('./blurred/imageBlurred2.jpg', imgBlurred2)
cv2.imwrite('./blurred/imageBlurred3.jpg', imgBlurred3)

cv2.imwrite('./deriatives/imageX.jpg', imgDerivativesX)
cv2.imwrite('./deriatives/imageY.jpg', imgDerivativesY)

cv2.imwrite('./deriatives/imageBlurred1X.jpg', imgDerivativesX1)
cv2.imwrite('./deriatives/imageBlurred1Y.jpg', imgDerivativesY1)

cv2.imwrite('./deriatives/imageBlurred2X.jpg', imgDerivativesX2)
cv2.imwrite('./deriatives/imageBlurred2Y.jpg', imgDerivativesY2)

cv2.imwrite('./deriatives/imageBlurred3X.jpg', imgDerivativesX3)
cv2.imwrite('./deriatives/imageBlurred3Y.jpg', imgDerivativesY3)

cv2.imwrite('./magnitude/image.jpg', imgMagnitude)
cv2.imwrite('./magnitude/imageBlurred1.jpg', imgMagnitude1)
cv2.imwrite('./magnitude/imageBlurred2.jpg', imgMagnitude2)
cv2.imwrite('./magnitude/imageBlurred3.jpg', imgMagnitude3)

close = False
while(not close):
    key = cv2.waitKey(0) & 0xFF
    if key == 27 or key == 255:
        close = True
    elif key == ord('n'):
        cv2.imshow('Image', img)
    elif key == ord('g'):
        while(not close):
            key = cv2.waitKey(0) & 0xFF
            if key == 27 or key == 255:
                close = True
            elif key == ord('1'):
                cv2.imshow('Image', img)
            elif key == ord('2'):
                cv2.imshow('Image', imgBlurred1)
            elif key == ord('3'):
                cv2.imshow('Image', imgBlurred2)
            elif key == ord('4'):
                cv2.imshow('Image', imgBlurred3)
        if key == 27:
            close = False
            cv2.imshow('Image', img)
    elif key == ord('d'):
        while(not close):
            key = cv2.waitKey(0) & 0xFF
            if key == 27 or key == 255:
                close = True
            elif key == ord('1'):
                cv2.imshow('Image', imgDerivativesX)
            elif key == ord('2'):
                cv2.imshow('Image', imgDerivativesY)
            elif key == ord('3'):
                cv2.imshow('Image', imgDerivativesX1)
            elif key == ord('4'):
                cv2.imshow('Image', imgDerivativesY1)
            elif key == ord('5'):
                cv2.imshow('Image', imgDerivativesX2)
            elif key == ord('6'):
                cv2.imshow('Image', imgDerivativesY2)
            elif key == ord('7'):
                cv2.imshow('Image', imgDerivativesX3)
            elif key == ord('8'):
                cv2.imshow('Image', imgDerivativesY3)
        if key == 27:
            close = False
            cv2.imshow('Image', img)
    elif key == ord('m'):
        while(not close):
            key = cv2.waitKey(0) & 0xFF
            if key == 27 or key == 255:
                close = True
            elif key == ord('1'):
                cv2.imshow('Image', imgMagnitude)
            elif key == ord('2'):
                cv2.imshow('Image', imgMagnitude1)
            elif key == ord('3'):
                cv2.imshow('Image', imgMagnitude2)
            elif key == ord('4'):
                cv2.imshow('Image', imgMagnitude3)
        if key == 27:
            close = False
            cv2.imshow('Image', img)
cv2.destroyAllWindows()
