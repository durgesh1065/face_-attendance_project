import cv2
import os

def capture_faces():
    name = input("Enter student name: ")
    roll = input("Enter roll number: ")
    path = f"dataset/{name}_{roll}"
    os.makedirs(path, exist_ok=True)
    
    cam = cv2.VideoCapture(0)
    count = 0
    
    while count < 30:  # Capture 30 samples
        ret, frame = cam.read()
        if not ret:
            break
            
        # Face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            face_img = frame[y:y+h, x:x+w]
            cv2.imwrite(f"{path}/{count}.jpg", face_img)
            count += 1
        
        cv2.imshow("Capturing", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_faces()