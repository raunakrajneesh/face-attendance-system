import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import sqlite3

# -------- SETTINGS --------
TOLERANCE = 0.5  # lower = stricter recognition

# Path to images
path = 'images'
images = []
classNames = []

# Load images
if os.path.exists(path):
    for cl in os.listdir(path):
        img = cv2.imread(f'{path}/{cl}')
        if img is not None:
            images.append(img)
            classNames.append(os.path.splitext(cl)[0])
else:
    print("Images folder not found!")
    exit()

if len(images) == 0:
    print("No images found!")
    exit()

# Encode faces
def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)
        if enc:
            encodeList.append(enc[0])
    return encodeList

encodeListKnown = find_encodings(images)
print("Encoding Complete")

# -------- DATABASE --------
def init_db():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            date TEXT,
            time TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')

    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM logs WHERE name=? AND date=?", (name, date))
    if cursor.fetchone() is None:
        cursor.execute("INSERT INTO logs (name, date, time) VALUES (?, ?, ?)",
                       (name, date, time))
        conn.commit()

    conn.close()

# -------- WEBCAM --------
print("Starting Face Recognition System...")
cap = cv2.VideoCapture(0)

# Reduce resolution (important for speed)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("Camera error")
    exit()

recognized_students = set()

process_this_frame = True
current_face_locations = []
current_face_names = []
current_face_colors = []
current_confidence = []

while True:
    success, img = cap.read()
    if not success:
        break

    if process_this_frame:
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_locations(imgS, model="hog")
        encodes = face_recognition.face_encodings(imgS, faces)

        current_face_locations = []
        current_face_names = []
        current_face_colors = []
        current_confidence = []

        for encodeFace, faceLoc in zip(encodes, faces):
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            name = "UNKNOWN"
            color = (0, 0, 255)
            confidence = 0

            if len(faceDis) > 0:
                matchIndex = np.argmin(faceDis)
                confidence = 1 - faceDis[matchIndex]

                if faceDis[matchIndex] < TOLERANCE:
                    name = classNames[matchIndex].upper()
                    color = (0, 255, 0)

                    # Prevent repeated DB calls
                    if name not in recognized_students:
                        recognized_students.add(name)
                        mark_attendance(name)

            current_face_locations.append(faceLoc)
            current_face_names.append(name)
            current_face_colors.append(color)
            current_confidence.append(confidence)

    process_this_frame = not process_this_frame

    # -------- UI --------
    for faceLoc, name, color, conf in zip(
        current_face_locations,
        current_face_names,
        current_face_colors,
        current_confidence
    ):
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, -1)

        text = f"{name} ({conf:.2f})"
        cv2.putText(img, text, (x1 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    (255, 255, 255), 2)

    # Header
    width = img.shape[1]
    cv2.rectangle(img, (0, 0), (width, 50), (30, 30, 30), -1)

    cv2.putText(img, "Face Attendance System",
                (20, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1,
                (0, 200, 255), 2)

    cv2.putText(img, f"Count: {len(recognized_students)}",
                (width - 200, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1,
                (0, 255, 150), 2)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()