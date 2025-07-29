import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Choose mode : ")
print("1. Real Time Webcam")
print("2. Video File ")
choice=input("Enter 1 or 2")

if choice=="1":
    capture=cv2.VideoCapture(0)
elif choice=="2":
    video_path=input("Enter the path of the Video File")
    capture=cv2.VideoCapture(video_path)
else:
    print("Invalid Choice ")
    exit()

if not capture.isOpened():
    print("Error. Cannot open video source")
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        print("Camera or Video Error")
        break

    frame = cv2.resize(frame, (800, 600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

   
    total_detected = len(faces)
    cv2.putText(frame, f"Faces Detected: {total_detected}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    
    cv2.imshow("", frame)    
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break


capture.release()
cv2.destroyAllWindows()


