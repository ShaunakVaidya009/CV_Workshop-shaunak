import cv2
import numpy as np
#load image
image= cv2.imread("C:/Users/DELL/Desktop/open CV/Day 1/sample.jpg")
resized_image = cv2.resize(image,(300,300))
average_blur = cv2.blur(resized_image,(7,7))
gaussion_blur= cv2.GaussianBlur(resized_image,(7,7),0)
median_blur= cv2.medianBlur(resized_image,7)
bilateral_blur = cv2.bilateralFilter(resized_image, 15,75,75)


cv2.imshow('Original Image',resized_image)
cv2.imshow('Average Image',average_blur)
cv2.imshow('Gaussion Image',gaussion_blur)
cv2.imshow('Median Image',median_blur)
cv2.imshow('Bilateral Image',bilateral_blur)



cv2.waitKey(0)
cv2.destroyAllWindows()