# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 21:59:00 2024

@author: saaml
"""
 

import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import face_recognition

# Initialize webcam
video_capture = cv2.VideoCapture("dataser/video.mp4")

#   Model Initialization
face_exp_model = load_model("dataset/facial_expression_model_combined.h5")

#   List of Emotions label
emotions_label = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")



r = 0
while True:
    
    r += 1
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break
    
    frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    
    # Convert the frame from BGR (OpenCV format) to RGB (face_recognition format)
    #rgb_frame = frame_small[:, :, ::-1]

    # Find all face locations in the current frame
    all_face_locations = face_recognition.face_locations(frame_small, model='cnn')

    # Draw rectangles around each detected face
    for top, right, bottom, left in all_face_locations:
        
        top *- 4
        right *= 4
        bottom *= 4
        left *=  4
        
        face_area = frame[top:bottom, left:right]
      
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        #   Preprocess image, convert it to an image as the data in the dataset
        
        #   Convert to Grayscale
        face_area = cv2.cvtColor(face_area, cv2.COLOR_BGR2GRAY)
        
        #   Resize to 48 x 48 pixels
        face_area = cv2.resize(face_area, (48, 48))
        
        #   Convert the PIL image into a 3D numpy array
        img_pixels = image.img_to_array(face_area)
        
        #   expand the shape of an array into single row multiple columns
        img_pixels = np.expand_dims(img_pixels, axis=0)
        
        #   Pixels are in range of [0, 255]. Normalize all pixels in scale of [0,1]
        #   img_pixels /= 255
        
        #   Predecting values for all 7 expressions
        exp_predic = face_exp_model.predict(img_pixels)
        
        #   Find max indexed prediction value(0 till 7)
        max_index = np.argmax(exp_predic[0])
        
        #   Get corresponding lable from emotion label
        emotion_label = emotions_label[max_index]
        
        #   Display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, emotion_label, (left, bottom), font, 0.5, (255, 123, 231), 1)
    
        

    if r % 10 == 0:
        print('There are {} number faces of this image' .format(len(all_face_locations)))
    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
