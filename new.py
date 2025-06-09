# capture_faces.py
import cv2
import os

def capture_faces():
    name = input("Enter student name: ")
    roll = input("Enter roll number: ")
    path = f"dataset/{name}_{roll}"
    os.makedirs(path, exist_ok=True)

    cam = cv2.VideoCapture(0)
    count = 0

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while count < 30:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            cv2.imwrite(f"{path}/{count}.jpg", face_img)
            count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_faces()


# face_train.py
import face_recognition
import pickle
import os

def train_model():
    known_encodings = []
    known_details = []

    for person_dir in os.listdir("dataset"):
        name, roll = person_dir.split("_")
        person_path = os.path.join("dataset", person_dir)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_details.append({"name": name, "roll": roll})

    os.makedirs("encodings", exist_ok=True)
    with open("encodings/face_encodings.pkl", "wb") as f:
        pickle.dump({"encodings": known_encodings, "details": known_details}, f)

if __name__ == "__main__":
    train_model()


# main.py
import face_recognition
import cv2
import numpy as np
import pickle
import csv
from datetime import datetime
import os

with open("encodings/face_encodings.pkl", "rb") as f:
    data = pickle.load(f)

video_capture = cv2.VideoCapture(0)
attended = set()
os.makedirs("attendance", exist_ok=True)

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(data["encodings"], face_encoding)
        face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
        best_match_index = np.argmin(face_distances)

        name = "Unknown"
        roll = ""

        if matches[best_match_index]:
            student = data["details"][best_match_index]
            name = student["name"]
            roll = student["roll"]

            if roll not in attended:
                attended.add(roll)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_file = f"attendance/{datetime.now().strftime('%Y%m%d')}.csv"
                with open(csv_file, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, roll, timestamp])

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} - {roll}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Face Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


# requirements.txt
opencv-python
face_recognition
numpy
pandas
python-csv
datetime
