import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height, width = image.shape
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def make_coordinates(image, line_parameters):
    if line_parameters is not None:
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (3/5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])
    else:
        return None

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0) if left_fit else None
    right_fit_average = np.average(right_fit, axis=0) if right_fit else None

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    if left_line is not None and right_line is not None:
        return np.array([left_line, right_line])
    else:
        return np.array([])  


def find_lane_center(image, lines):
    if lines is not None and len(lines) >= 2:
        line1 = lines[0]
        line2 = lines[1]
        x1_1, y1_1, x2_1, y2_1 = line1
        x1_2, y1_2, x2_2, y2_2 = line2
        distance = abs((x1_2 + x2_2) / 2 - (x1_1 + x2_1) / 2)
        middle_point_x = int((x1_1 + x2_1 + x1_2 + x2_2) / 4)
        middle_point_y = int((y1_1 + y2_1 + y1_2 + y2_2) / 4)

        return distance, (middle_point_x, middle_point_y)
    else:
        return None, None

cap = cv2.VideoCapture("test2.mp4")

while cap.isOpened():
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    if lines is not None:
        averaged_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        
        distance, center_point = find_lane_center(frame, averaged_lines)
        
        if distance is not None and center_point is not None:
            cv2.putText(frame, f"Distance Between Lanes: {distance}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.circle(frame, center_point, 5, (0, 0, 255), -1)
        
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("result", combo_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
