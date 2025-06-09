from flask import Flask, render_template, request, redirect, Response
import cv2
import os
import face_recognition
import pickle
from datetime import datetime
import csv
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    if request.method == 'POST':
        name = request.form['name']
        roll = request.form['roll']
        path = f"dataset/{name}_{roll}"
        os.makedirs(path, exist_ok=True)

        count = 0
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while count < 30:
            ret, frame = camera.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]
                save_path = f"{path}/{count}.jpg"
                cv2.imwrite(save_path, face_img)
                count += 1
                print(f"[INFO] Saved {save_path}")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Enrolling - Press 'q' to exit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        return redirect('/train')
    return render_template('enroll.html')

@app.route('/train')
def train_model():
    known_encodings = []
    known_details = []

    for person_dir in os.listdir("dataset"):
        if "_" not in person_dir:
            continue
        name, roll = person_dir.split("_", 1)
        for image_file in os.listdir(f"dataset/{person_dir}"):
            image_path = f"dataset/{person_dir}/{image_file}"
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_details.append({"name": name, "roll": roll})

    os.makedirs("encodings", exist_ok=True)
    with open("encodings/face_encodings.pkl", "wb") as f:
        pickle.dump({"encodings": known_encodings, "details": known_details}, f)

    return redirect('/attendance')

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

def gen_frames():
    with open("encodings/face_encodings.pkl", "rb") as f:
        data = pickle.load(f)

    attended = set()

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, location in zip(face_encodings, face_locations):
            name, roll = "Unknown", ""
            distances = face_recognition.face_distance(data["encodings"], face_encoding)

            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                if distances[best_match_index] < 0.5:
                    student = data["details"][best_match_index]
                    name, roll = student["name"], student["roll"]

                    if roll not in attended:
                        attended.add(roll)
                        os.makedirs("attendance", exist_ok=True)
                        with open(f"attendance/{datetime.now().strftime('%Y%m%d')}.csv", "a", newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([name, roll, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

            top, right, bottom, left = location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} - {roll}", (left + 6, bottom + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
