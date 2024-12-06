import cv2
import os

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# User ID for saving images
user_id = input("Enter user ID: ")
name = input("Enter user name: ")
print("Capturing images... Look at the camera.")

# Ensure the dataset directory exists
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Capture and save 30 face samples
count = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        count += 1
        # Save the captured image into the dataset folder
        cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg", gray[y:y+h, x:x+w])
        
        # Draw rectangle around the face and display the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Face Capture', frame)
    
    # Break if enough images are taken or 'q' is pressed
    if count >= 30 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Images for {name} with ID {user_id} captured successfully.")

