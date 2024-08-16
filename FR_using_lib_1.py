# -*- coding: utf-8 -*-
# import cmake
import cv2
import face_recognition
# image_to_detect = cv2.imread('myself_1.jpg')
image_to_detect = cv2.imread('MongoDB_Training.jpeg')

cv2.imshow('hello', image_to_detect)
cv2.waitKey(0)
cv2.destroyAllWindows()

all_face_locations = face_recognition.face_locations(image_to_detect, model='hog')

print('There are {} number faces of this image' .format(len(all_face_locations)))

for index, current_face_location in enumerate(all_face_locations):
    top_pos, right_pos, bot_pos, left_pos = current_face_location
    print("Found faces {} at top:{}, right:{}, bottom:{}, left:{}" .format(index+ 1, top_pos, right_pos, bot_pos, left_pos ))
    current_face_image = image_to_detect[top_pos:bot_pos, left_pos:right_pos]
    current_face_image = cv2.resize(current_face_image, (200, 200))
    cv2.imshow('Face No: '  +str(index), current_face_image)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    #cv2.rectangle(image_to_detect, (left_pos, top_pos), (right_pos, bot_pos), (0, 255, 0), 2)
    
cv2.imshow('hello', image_to_detect)
cv2.waitKey(0)
cv2.destroyAllWindows()