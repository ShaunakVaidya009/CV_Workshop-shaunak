import cv2
import numpy as np
#load image
image= cv2.imread("C:/Users/DELL/Desktop/open CV/Day 1/sample.jpg")
resized_image = cv2.resize(image,(300,300))
average_blur = cv2.blur(resized_image,(7,7))
gaussion_blur= cv2.GaussianBlur(resized_image,(7,7),0)
median_blur= cv2.medianBlur(resized_image,7)
bilateral_blur = cv2.bilateralFilter(resized_image, 15,75,75)


def put_label(img,text,pos=(10,25)):
    return cv2.putText(img, text,pos, cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

#labeling
original_lbl =put_label(resized_image.copy(),"Original Image")
average_lbl = put_label(resized_image.copy(),"Average Image")
gaussion_lbl =put_label(resized_image.copy(),"Gaussion Image")
median_lbl =put_label(resized_image.copy(),"Median Image")
bilateral_lbl =put_label(resized_image.copy(),"Bilateral Image")

#DIsplay
row1 = np.hstack((original_lbl,average_lbl,gaussion_lbl  ))
row2 = np.hstack((median_lbl,bilateral_lbl, np.zeros_like(resized_image))) 

final= np.vstack((row1,row2))
cv2.imshow('Final Blurred Images',final)


# cv2.imshow('Original Image',resized_image)
# cv2.imshow('Average Image',average_blur)
# cv2.imshow('Gaussion Image',gaussion_blur)
# cv2.imshow('Median Image',median_blur)
# cv2.imshow('Bilateral Image',bilateral_blur)



cv2.waitKey(0)
cv2.destroyAllWindows()