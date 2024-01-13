import cv2
cap = cv2.VideoCapture('C:/Users/BurNet/Desktop/finding-lanes/arabayakala/cars.mp4')

car_cascade = cv2.CascadeClassifier('C:/Users/BurNet/Desktop/finding-lanes/arabayakala/cars.xml')

if car_cascade.empty():
    print("XML dosyası bulunamadı.")
    exit()

while True:
    ret, frame = cap.read()
    cars = car_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'X: {x}, Y: {y}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
