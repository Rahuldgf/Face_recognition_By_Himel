import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = 'C:/Users/DeepOP/OneDrive/Desktop/dataset/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [], []

for i, file in enumerate(onlyfiles):
    image_path = data_path + file
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(Training_Data, Labels)

print("Dataset model training completed")

face_classifier = cv2.CascadeClassifier(
    "C:/Users/DeepOP/Downloads/haarcascade_frontalface_default.xml"
)

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is None or len(faces) == 0:
        return img, []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = img[y : y + h, x : x + w]
        roi = cv2.resize(roi, (200, 200))
        return img, roi

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        confidence = int(100 * (1 - result[1] / 300))
        if confidence > 82:
            cv2.putText(
                image,
                "Himel Dutta",
                (250, 450),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2,
            )
        else:
            cv2.putText(
                image,
                "Unknown",
                (250, 450),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        cv2.imshow("Face Cropper", image)
    except Exception:
        cv2.putText(
            image,
            "Face not found",
            (250, 450),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        cv2.imshow("Face Cropper", image)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()


