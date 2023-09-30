import cv2
import joblib
import os
import face_recognition

model = joblib.load('face_model.pkl')

cap = cv2.VideoCapture(0)

while True:
    try:
        ret, image = cap.read()

        image_loc = face_recognition.face_locations(image)
        print(image_loc)
        for i in range(len(image_loc)):

            encoding = face_recognition.face_encodings(image)[i]

            result = model.predict([encoding])

            left = (image_loc[i][3], image_loc[i][0])
            right = (image_loc[i][1], image_loc[i][2] + 20)
            color = [0, 0, 255]
            cv2.rectangle(image, left, right, color, 4, cv2.FILLED)
            cv2.putText(
                image,
                result[0],
                (image_loc[i][3] + 10, image_loc[i][2]+15),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 0, 0),
                4
            )
    except:
        continue

    cv2.imshow("Video", image)

    k = cv2.waitKey(20)
    if k == ord('q'):
        break

