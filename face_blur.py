# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 20:31:05 2024

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
    # frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the frame from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = frame[:, :, ::-1]

    # Find all face locations in the current frame
    all_face_locations = face_recognition.face_locations(rgb_frame, model='hog')

    # Draw rectangles around each detected face
    for top, right, bottom, left in all_face_locations:
        # top *- 4
        # right *= 4
        # bottom *= 4
        # left *=  4
        
        # Blurring part of the code
        face_area = frame[top:bottom, left:right]
        face_area = cv2.GaussianBlur(face_area, (99, 99), 20)
        frame[top:bottom, left:right] = face_area
        
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
