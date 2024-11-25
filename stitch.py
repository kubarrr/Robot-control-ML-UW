import cv2
import matplotlib.pyplot as plt
import numpy as np

############################
########## TASK 1 ########## 
############################

img = cv2.imread(f"./calibration/img1.png")
size=(img.shape[1], img.shape[0])

calibration_images = []
for i in range(1, 28):
    img = cv2.imread(f"./calibration/img{i}.png")
    calibration_images.append(img)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# First calibration method
def creating_one_object_points():
    square_size = 168
    spacing=70
    one_object_point=np.array([[0, 0, 0], [0, -square_size, 0], [square_size, -square_size, 0], [square_size, 0, 0]], dtype=np.float32)
    object_points=[]
    for i in range(2):
        for j in range(3):
            offset_x = j * (square_size + spacing)
            offset_y = -i * (square_size + spacing)
            object_points.append(one_object_point + [offset_x, offset_y, 0])
    object_points = np.array(object_points, dtype=np.float32)
    object_points=object_points.reshape(-1, 3)
    return object_points

def calibrate_camera_all_board():
    img_points = [] 
    obj_points = [] 
    one_object_points=creating_one_object_points()

    for img in calibration_images:
        corners, ids, _ = detector.detectMarkers(img)
        
        corners=np.array(corners)
        desired_order = [29, 24, 19, 28, 23, 18]

        ids_flat = ids.flatten()
        id_corner_map = {id: corner for id, corner in zip(ids_flat, corners)}
        sorted_corners = [id_corner_map[id] for id in desired_order]
        sorted_corners=np.array(sorted_corners)
        img_points.append(sorted_corners)
        obj_points.append(one_object_points)

    obj_points=np.array(obj_points)
    img_points = np.array(img_points, dtype=np.float32)
    img_points=img_points.reshape(27, 24, 2)

    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    return ret, cameraMatrix, distCoeffs, rvecs, tvecs, obj_points, img_points

#Undistorting
def undistort_image(img, cameraMatrix, distCoeffs):
    size = (img.shape[1], img.shape[0])
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, size, alpha=0)
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, np.eye(3), new_camera_matrix, size, m1type=cv2.CV_32FC1)
    undistorted_image = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    return undistorted_image

# ret, cameraMatrix, distCoeffs, rvecs, tvecs, obj_points, img_points  = calibrate_camera_all_board()
# img_to_undistort=calibration_images[0]
# undistorted_image = undistort_image(img_to_undistort, cameraMatrix, distCoeffs)
# comparasion = np.hstack((img_to_undistort, undistorted_image))
# cv2.imshow("Distorted vs Undistorted", comparasion)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#Second calibration method
def calibrate_camera_one_marker():
    square_size = 168
    one_object_points2=np.array([[0, 0, 0], [0, -square_size, 0], [square_size, -square_size, 0], [square_size, 0, 0]])
    img_points2=[]
    obj_points2=[]
    for img in calibration_images:
        corners, ids, _ = detector.detectMarkers(img)
        corners=np.array(corners)
        for i in range(len(corners)):
            img_points2.append(corners[i])
            obj_points2.append(one_object_points2)
    obj_points2=np.array(obj_points2, dtype=np.float32)
    img_points2=np.array(img_points2)
    img_points2=img_points2.reshape(162, 4, 2)

    ret2, cameraMatrix2, distCoeffs2, rvecs2, tvecs2 = cv2.calibrateCamera(obj_points2, img_points2, size, None, None)
    return ret2, cameraMatrix2, distCoeffs2, rvecs2, tvecs2, obj_points2, img_points2

# ret2, cameraMatrix2, distCoeffs2, rvecs2, tvecs2, obj_points2, img_points2 = calibrate_camera_one_marker()
# img_to_undistort=calibration_images[0]
# undistorted_image2 = undistort_image(img_to_undistort, cameraMatrix2, distCoeffs2)
# comparasion2 = np.hstack((img_to_undistort, undistorted_image2))
# cv2.imshow("Distorted vs Undistorted", comparasion2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print("Calibration with known distances between markers")
# print(cameraMatrix)
# print(distCoeffs)
# print("Calibration based on one marker")
# print(cameraMatrix2)
# print(distCoeffs2)

def calculate_reprojection_error(obj_points, img_points, rvecs, tvecs, cameraMatrix, distCoeffs):
    mean_error = 0
    for i in range(len(obj_points)):
        imgpoints_projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
        imgpoints_projected = imgpoints_projected.reshape(-1, 2)
        error = cv2.norm(img_points[i], imgpoints_projected, cv2.NORM_L2)/len(imgpoints_projected)
        mean_error += error
    return mean_error

# print(f"Error for first method is {calculate_reprojection_error(obj_points, img_points, rvecs, tvecs, cameraMatrix, distCoeffs)/len(obj_points)}")
# print(f"Error for second method is {calculate_reprojection_error(obj_points2, img_points2, rvecs2, tvecs2, cameraMatrix2, distCoeffs2)/len(obj_points2)}")

############################
########## TASK 2 ########## 
############################

def apply_projective_transformation(src_image, H):
    height = src_image.shape[0]
    width = src_image.shape[1]
    invH = np.linalg.inv(H)

    corners = np.array([[0, 0, 1], [width-1, 0, 1], [0, height-1, 1], [width-1, height-1, 1]])
    transformed_corners = np.matmul(H, corners.T).T
    transformed_corners /= transformed_corners[:, 2].reshape(-1, 1)
    
    min_x = int(np.floor(min(transformed_corners[:, 0])))
    max_x = int(np.ceil(max(transformed_corners[:, 0])))
    min_y = int(np.floor(min(transformed_corners[:, 1])))
    max_y = int(np.ceil(max(transformed_corners[:, 1])))

    new_width = max_x - min_x
    new_height = max_y - min_y

    dst_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            dst_pixel = np.array([x+min_x, y+min_y, 1])
            src_pixel = np.matmul(invH, dst_pixel)
            src_x, src_y = int(round(src_pixel[0] / src_pixel[2])), int(round(src_pixel[1] / src_pixel[2]))
            if 0 <= src_x < width and 0 <= src_y < height:
                dst_image[y, x] = src_image[src_y, src_x]
                    
    cv2.imshow("orginal", src_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("transforem", dst_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return dst_image, min_x, max_x, min_y, max_y

# src_img = cv2.imread(f"./stitching/img1.png")
# cv2.imshow("example source stitching image", src_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# dst_img = cv2.imread(f"./stitching/img2.png")
# cv2.imshow("example destination stitching image", dst_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# H=np.array([[ 7.98740818e-01, -2.10795707e-02,  1.69309897e+02],
#        [-4.99802327e-02,  8.80379806e-01,  4.50679763e+01],
#        [-1.33520799e-04, -3.90246481e-05,  1.00000000e+00]])

# apply_projective_transformation(src_img, H)

############################
########## TASK 3 ########## 
############################

def find_homography(src_pts, dst_pts):
    A=[]
    for i in range(len(src_pts)):
        A.append([src_pts[i][0], src_pts[i][1], 1, 0, 0, 0, -dst_pts[i][0]*src_pts[i][0], -dst_pts[i][0]*src_pts[i][1], -dst_pts[i][0]])
        A.append([0, 0, 0, src_pts[i][0], src_pts[i][1], 1, -dst_pts[i][1]*src_pts[i][0], -dst_pts[i][1]*src_pts[i][1], -dst_pts[i][1]])
    A=np.array(A)
    M=np.transpose(A)
    _, _, V = np.linalg.svd(A)
    eigenvector = V[-1, :]
    #we assume h33=1
    eigenvector=eigenvector/eigenvector[-1]
    H=eigenvector.reshape(3, 3)
    return H

def test_find_homography(H, shape=[720, 1280]):
    H=H/H[-1, -1]
    n=5
    for i in range(n):
        number_of_points=4
        src_points_y=np.random.randint(shape[0], size=(number_of_points, 1))
        src_points_x=np.random.randint(shape[1], size=(number_of_points, 1))
        src_points=np.hstack((src_points_x, src_points_y, np.ones((number_of_points, 1))))
        dst_points=np.matmul(H, src_points.T).T
        z=dst_points[:, 2]
        z=z.reshape(-1, 1)
        dst_points=dst_points/z
        H_found = find_homography(src_points, dst_points)
        assert np.allclose(H_found, H, rtol=1e-3), "failed"
        print(f" test passed")

random_homographies=[np.array([[1, 0, 120], [0, 1, 1], [0, 0, 1]]), np.array([[1, 1, 1], [1, 1.5, 1], [0, 0.1, 1.5]]), np.array([[0.2, 0.23, 100],[0.01, 0.01, 200],[0, 0, 0.7]])]
for H in random_homographies:
    test_find_homography(H)

############################
########## TASK 4 ########## 
############################

# src_img = cv2.imread(f"./stitching/img1.png")
# cv2.imshow("example source stitching image", src_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# dst_img = cv2.imread(f"./stitching/img2.png")
# cv2.imshow("example destination stitching image", dst_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# pts=[]
# def get_coordinates(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
#         pts.append([x, y])
#         print(f"Coordinates: x: {x}, y: {y})")

# zoomed_src_img=src_img
# zoomed_src_img=src_img[200:600, 200:600]
# zoomed_src_img = cv2.resize(zoomed_src_img, (1000, 1000))
# cv2.imshow("example zoomed source image", zoomed_src_img)
# cv2.setMouseCallback('example zoomed source image', get_coordinates)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# def correct_x(x, start, end, width):
#     return round(start+(x/width)*(end-start))
# def correct_y(y, start, end, height):
#     return round(start+(y/height)*(end-start))

# src_pts=np.array(pts)
# src_pts_corrected=np.array(src_pts)
# for i in range(len(src_pts)):
#     src_pts_corrected[i][0]=correct_x(src_pts_corrected[i][0], 200, 600, 1000)
#     src_pts_corrected[i][1]=correct_y(src_pts_corrected[i][1], 200, 600, 1000)

# pts=[]
# zoomed_dst_img=dst_img[200:600, 400:800]
# zoomed_dst_img = cv2.resize(zoomed_dst_img, (1000, 1000))
# cv2.imshow("example zoomed destination image", zoomed_dst_img)
# cv2.setMouseCallback('example zoomed destination image', get_coordinates)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# dst_pts=np.array(pts)
# dst_pts_corrected=np.array(dst_pts)
# for i in range(len(dst_pts)):
#     dst_pts_corrected[i][0]=correct_x(dst_pts_corrected[i][0], 400, 800, 1000)
#     dst_pts_corrected[i][1]=correct_y(dst_pts_corrected[i][1], 200, 600, 1000)

# H=find_homography(src_pts_corrected, dst_pts_corrected)
H=np.array([[ 7.98740818e-01, -2.10795707e-02,  1.69309897e+02],
       [-4.99802327e-02,  8.80379806e-01,  4.50679763e+01],
       [-1.33520799e-04, -3.90246481e-05,  1.00000000e+00]])
# apply_projective_transformation(src_img, H)

    




