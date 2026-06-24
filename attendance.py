import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

# Dataset Folder
path = "dataset"

images = []
classNames = []

myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])


# Face Encoding
def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 0:
            encodeList.append(encodings[0])

    return encodeList


# Attendance Function
def markAttendance(name):

    file_path = "attendance/Attendance.csv"

    today = datetime.now().strftime("%d-%m-%Y")
    current_time = datetime.now().strftime("%H:%M:%S")

    lines_to_keep = ["Name,Date,Time\n"]

    try:
        with open(file_path, "r") as f:

            lines = f.readlines()

            for line in lines[1:]:

                data = line.strip().split(",")

                if len(data) >= 3:

                    attendance_date = data[1]

                    if attendance_date == today:
                        lines_to_keep.append(line)

    except FileNotFoundError:
        pass

    already_marked = False

    for line in lines_to_keep[1:]:

        data = line.strip().split(",")

        if len(data) >= 3:

            if data[0] == name:
                already_marked = True
                break

    if not already_marked:

        with open(file_path, "w") as f:

            f.writelines(lines_to_keep)

            f.write(f"{name},{today},{current_time}\n")

        print(f"{name} Attendance Saved")

    else:
        print(f"{name} Already Marked Today")


# Encode Faces
encodeListKnown = findEncodings(images)

print("Encoding Complete")

# Webcam
cap = cv2.VideoCapture(0)

attendance_marked = False

while True:

    success, img = cap.read()

    if not success:
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)

    encodesCurFrame = face_recognition.face_encodings(
        imgS,
        facesCurFrame
    )

    for encodeFace, faceLoc in zip(
        encodesCurFrame,
        facesCurFrame
    ):

        matches = face_recognition.compare_faces(
            encodeListKnown,
            encodeFace
        )

        faceDis = face_recognition.face_distance(
            encodeListKnown,
            encodeFace
        )

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:

            name = classNames[matchIndex].upper()

            markAttendance(name)

            attendance_marked = True

            y1, x2, y2, x1 = faceLoc

            y1, x2, y2, x1 = (
                y1 * 4,
                x2 * 4,
                y2 * 4,
                x1 * 4
            )

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            break

    cv2.imshow("Face Attendance System", img)

    if attendance_marked:
        cv2.waitKey(2000)
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()