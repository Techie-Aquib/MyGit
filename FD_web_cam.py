# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:49:36 2024

@author: saaml
"""

import cv2
import face_recognition

# Initialize webcam
video_capture = cv2.VideoCapture(0)

r = 0
while True:
    
    r += 1
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the frame from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = frame_small[:, :, ::-1]

    # Find all face locations in the current frame
    all_face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2, model='hog')

    # Draw rectangles around each detected face
    for top, right, bottom, left in all_face_locations:
        top *- 4
        right *= 4
        bottom *= 4
        left *=  4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

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
