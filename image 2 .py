import cv2
#load image
image= cv2.imread("C:/Users/DELL/Desktop/open CV/Day 1/sample.jpg")
resized_image = cv2.resize(image,(300,300))
gray_scale = cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)
hsv_image =cv2.cvtColor(resized_image,cv2.COLOR_BGR2HSV) 
rgb_image =cv2.cvtColor(resized_image,cv2.COLOR_BGR2RGB)
rotate_image = cv2.rotate(resized_image,cv2.ROTATE_180)
crop_image = resized_image[100:300,200:400]
flip_image = cv2.flip(resized_image,1)
cv2.imshow('Original Image',resized_image)
cv2.imshow('Gray_Scale Image',gray_scale)
cv2.imshow('Original Image',hsv_image)
cv2.imshow('RGB IMAGE',rgb_image)
cv2.imshow('Rotated Image',rotate_image)
cv2.imshow('Cropped Image',crop_image)
cv2.imshow('Flipped Image',flip_image)

cv2.waitKey(0)
cv2.destroyAllWindows()