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
    cv2.imshow("transformed", dst_image)
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


############################
########## TASK 5 ########## 
############################

src_img = cv2.imread(f"./stitching/img1.png")
# cv2.imshow("example source image", src_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

dst_img = cv2.imread(f"./stitching/img2.png")
# cv2.imshow("example destination image", dst_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def get_new_shape_dst(dst_img, src_img, offset_x, offset_y):
    height_src, width_src= src_img.shape[0], src_img.shape[1]
    height_dst, width_dst= dst_img.shape[0], dst_img.shape[1]

    offset_x=offset_x
    offset_y=offset_y
   
    size_x=width_src
    size_y=height_src
    new_shape = np.zeros((size_y, size_x, 3), dtype=np.uint8)
    # print(new_shape.shape)
    # print(src_img.shape)
    if offset_x>0:
        new_shape[-offset_y:height_dst-offset_y, :width_dst, :] = dst_img[:, :, :]
    return new_shape

src_img_transformed, offset_x, max_x, offset_y, max_y=apply_projective_transformation(src_img, H)
dst_reshaped=get_new_shape_dst(dst_img, src_img_transformed, offset_x, offset_y)
cv2.imshow("dst reshaped image", dst_reshaped)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("src transformed image", src_img_transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()

def calc_cost(pixel1, pixel2):
    diff = np.abs(pixel1 - pixel2)
    grayscale = 0.3 * diff[0] + 0.59 * diff[1] + 0.11 * diff[2]
    return grayscale ** 2

def find_path(src_img_transformed, dst_img, offset_x, offset_y, shape_y):
    height_dst, width_dst= dst_img.shape[0], dst_img.shape[1]
    height_src, width_src= src_img_transformed.shape[0], src_img_transformed.shape[1]
    if offset_x>0:
        src_img_overlap=src_img_transformed[-offset_y:-offset_y+shape_y, :width_dst-offset_x]
        dst_img_overlap=dst_img[-offset_y:-offset_y+shape_y, offset_x:]
        row_length=width_dst-offset_x
    else:
        src_img_overlap=src_img_transformed[-offset_y:-offset_y+shape_y, -offset_x:width_dst]
        dst_img_overlap=dst_img[-offset_y:-offset_y+shape_y, :width_dst+offset_x]
        row_length=width_dst+offset_x
    
    col_length=shape_y
    

    cv2.imshow("src image", src_img_overlap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("dst image", dst_img_overlap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #x_min, y_min, x_max, y_max = 169, 0, 1279, 719
    #col_length=height_src
    cost_tab=np.zeros((col_length, row_length))
    path_ids=np.zeros((col_length, row_length))
    for i in range(row_length):
        cost_tab[0, i]=calc_cost(src_img_overlap[0, i], dst_img_overlap[0, i])
    for i in range(1, col_length):
        for j in range(row_length):
            best_id=0
            best_cost=1000000
            for k in range(-1, 2, 1):
                if 0<=j+k<row_length:
                    prev_cost=cost_tab[i-1, j+k]
                    cost=calc_cost(src_img_overlap[i, j], dst_img_overlap[i, j])
                    cost_sum=prev_cost+cost
                    if cost_sum<best_cost:
                        best_id=k
                        best_cost=cost_sum
            cost_tab[i, j]=best_cost
            path_ids[i, j]=best_id

    best_cost=1000000
    best_id=0
    best_path=[]
    for i in range(row_length):
        if cost_tab[col_length-1, i]<best_cost:
            best_cost=cost_tab[col_length-1, i]
            best_id=i
    best_path.append([col_length-1, best_id])
    for i in range(col_length-2, -1, -1):
        x=int(best_path[-1][1])
        best_id=path_ids[i+1, x]
        best_path.append([i, int(x+best_id)])
    
    id_high=best_path[-1][1]
    for i in range(height_dst+offset_y-shape_y):
        best_path.append([-i, id_high])

    id_low=best_path[0][1]
    for i in range(-offset_y):
        best_path.insert(0, [shape_y-1+i, id_low])
    
    for i in range(len(best_path)):
        best_path[i][0]=best_path[i][0]+height_dst+offset_y-shape_y
    return best_path

print(offset_x, offset_y, size[1])
best_seam=find_path(src_img_transformed, dst_reshaped, offset_x, offset_y, size[1])
print(best_seam)

def plot_seam(image1, image2, seam, offset_x):
    if offset_x>0:
        offset_x1=0
        offset_x2=offset_x
    else:
        offset_x1=-offset_x
        offset_x2=0
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.plot([s[1]+offset_x1 for s in seam], [s[0] for s in seam], color='red', linewidth=1)
    plt.title("source image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.plot([s[1]+offset_x2 for s in seam], [s[0] for s in seam], color='red', linewidth=1)
    plt.title("destination image")
    
    plt.show()

plot_seam(src_img_transformed, dst_reshaped, best_seam, offset_x)

def stitch_images(dst_reshaped, src_img_transformed, seam, offset_x):
    #height, width= src_img_transformed.shape[0], src_img_transformed.shape[1]
    #height, width= dst_reshaped.shape[0], dst_reshaped.shape[1]
    if offset_x>0:
        height, width= src_img_transformed.shape[0], src_img_transformed.shape[1]
        new_image=np.zeros((height, width+offset_x, 3), dtype=np.uint8)
        for i in range(height):
            new_image[i, :(seam[height-1-i][1]+offset_x)]=dst_reshaped[i, :(seam[height-1-i][1]+offset_x)]
            new_image[i, (seam[height-1-i][1]+offset_x):]=src_img_transformed[i, (seam[height-1-i][1]):]
    else:
        height, width= dst_reshaped.shape[0], dst_reshaped.shape[1]
        new_image=np.zeros((height, width-offset_x, 3), dtype=np.uint8)
        print(new_image.shape)
        for i in range(height):
            new_image[i, (seam[height-1-i][1]-offset_x):]=dst_reshaped[i, (seam[height-1-i][1]):]
            new_image[i, :(seam[height-1-i][1]-offset_x)]=src_img_transformed[i, :(seam[height-1-i][1]-offset_x)]
            #new_image[i, :(seam[height-1-i][1]-offset_x)]=src_img[i, :(seam[height-1-i][1]-offset_x)]
            # print(seam[i][1]-offset_x)
            # print((new_image[i, int(seam[i][1]-offset_x):]).shape)
            # print((dst_img[i, seam[i][1]:]).shape)
            #new_image[i, (seam[height-1-i][1]-offset_x):]=dst_img[i, (seam[height-1-i][1]):]
    return new_image

stitched_img=stitch_images(dst_reshaped, src_img_transformed, best_seam, offset_x)
cv2.imshow("stitched image", stitched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("task_5_stitched.jpg", stitched_img)



