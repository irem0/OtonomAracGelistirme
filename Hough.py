import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(0, 255, 0), thickness=3):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    roi_vertices = [(0, height), (width/2, height/2), (width, height)]
    roi_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))

    lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

    line_img = np.zeros_like(image)
    draw_lines(line_img, left_lines + right_lines) 

    result = cv2.addWeighted(image, 0.8, line_img, 1, 0)

    return result

cap = cv2.VideoCapture("test2.mp4")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    processed_frame = process_image(frame)

    cv2.imshow("Lane Detection", processed_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
