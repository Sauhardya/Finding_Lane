"""
Author- Sauhardya Singha
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
    """
    converted to greyscale as it will help in less computations as RGB values are large and will make the computations complicated
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Now we will reduce noise from our image
    # This is done by making use of Gaussian Blur, it will make use of 5x5 filter and average out the weights
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # We will use the Canny() method for edge detection, that is, sharp change in intensities (or gradients,
    # strong gradients->steep change, small gradient->shallow change)
    canny = cv2.Canny(blur, 50, 150)
    return canny
    # it should be noted that when we will apply the Canny method, the gaussian blur will be applied automatically


def region_of_interest(image):
    # getting height(y-axis) of the image
    height = image.shape[0]
    polygons = np.array([[
        (200, height), (1100, height), (550, 250)
    ]])
    # polygons is an array of polygons, here a single triangle
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    lane_only = cv2.bitwise_and(image, mask)
    return lane_only


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def make_coordinates(image, line_average_parameters):
    slope, intercept = line_average_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


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
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


# image = cv2.imread(r"C:\Users\ssrev\PycharmProjects_Self_Driving_Car_Virtual_Implementation\Finding_Lanes
# \test_image.jpg") lane_image=np.copy(image) canny_image=canny(lane_image) processed_image=region_of_interest(
# canny_image) lines=cv2.HoughLinesP(processed_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
# averaged_lines=average_slope_intercept(lane_image,lines) lineImage=display_lines(lane_image,averaged_lines)
# final_result_image=cv2.addWeighted(lane_image,0.8,lineImage,1,1) cv2.imshow("result",region_of_interest(canny))
# cv2.waitKey(0) plt.imshow(canny_image) plt.show() cv2.imshow("result",final_result_image) cv2.waitKey(0)

# to identify the lane, we will draw a triangle with coordinates (200,700) , (1100,700) , (550,250)
# refer to def region_of_interest


capture = cv2.VideoCapture(r"C:\Users\ssrev\PycharmProjects_Self_Driving_Car_Virtual_Implementation\Finding_Lanes\test2.mp4")
while capture.isOpened():
    _, frame = capture.read()
    canny_image = canny(frame)
    processed_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(processed_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    lineImage = display_lines(frame, averaged_lines)
    final_result_image = cv2.addWeighted(frame, 0.8, lineImage, 1, 1)
    cv2.imshow("final_result", final_result_image)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
capture.release()
cv2.destroyAllWindows()

