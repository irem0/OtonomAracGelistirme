import cv2
import numpy as np

cap = cv2.VideoCapture('trafficlight2.mp4')

if not cap.isOpened():
    print("Error: Couldn't open the video file.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video stream")
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    mask = cv2.bitwise_or(mask_red, mask_green)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)

            color = "Kırmızı" if np.sum(mask_red[y:y+h, x:x+w]) > np.sum(mask_green[y:y+h, x:x+w]) else "Yeşil"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Traffic Light Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
