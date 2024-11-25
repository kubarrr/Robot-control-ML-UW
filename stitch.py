import cv2
import matplotlib.pyplot as plt
import numpy as np


def calibrate_camera_all_board():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)


if __name__ == '__main__':
    img = cv2.imread(f"./calibration/img1.png")
    size=(img.shape[1], img.shape[0])
    cv2.imshow("img1", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
