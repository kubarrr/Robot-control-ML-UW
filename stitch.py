import cv2
import matplotlib.pyplot as plt
import numpy as np

# FUNCTIONS USED IN DIFFERENT TASKS

# APPLYING PROJECTIVE TRANSFORMATION
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

# FINDING HOMOGRAPHY
def find_homography(src_pts, dst_pts):
    A=[]
    for i in range(len(src_pts)):
        A.append([src_pts[i][0], src_pts[i][1], 1, 0, 0, 0, -dst_pts[i][0]*src_pts[i][0], -dst_pts[i][0]*src_pts[i][1], -dst_pts[i][0]])
        A.append([0, 0, 0, src_pts[i][0], src_pts[i][1], 1, -dst_pts[i][1]*src_pts[i][0], -dst_pts[i][1]*src_pts[i][1], -dst_pts[i][1]])
    A=np.array(A)
    M=np.transpose(A)
    _, _, V = np.linalg.svd(A)
    eigenvector = V[-1, :]
    # WE ASSUME h33=1
    eigenvector=eigenvector/eigenvector[-1]
    H=eigenvector.reshape(3, 3)
    return H

# RESHAPEING PHOTOS TO PLACE THEM ON WHOLE PANORAMA
def get_new_shape(img, min_x, max_x, min_y, max_y, offset_x, offset_y):
    height, width=img.shape[0], img.shape[1]
    up=-min_y
    down=max_y-height
    height_new=up+height+down
    left=-min_x
    right=max_x-width
    width_new=left+width+right
    shape_new = np.zeros((height_new, width_new, 3), dtype=np.uint8)
    print(shape_new.shape)
    shape_new[-offset_y:height-offset_y, offset_x:offset_x+width, :]=img[:, :, :]

    return shape_new

# CALCULATING COST WITH MODIFICATION TO PENALISE NOT OVERLAPPING REGIONS
def calc_cost(left, right):
    diff = np.abs(left - right)
    grayscale = 0.3 * diff[:, :, 0] + 0.59 * diff[:, :, 1] + 0.11 * diff[:, :, 2]
    pixel1_black=np.all(left == 0, axis=-1)
    pixel2_black = np.all(right == 0, axis=-1)
    not_overlap = (pixel1_black & ~pixel2_black) | (~pixel1_black & pixel2_black)
    penalty=grayscale**2+not_overlap*1000000000
    return penalty

# FINDING SEAM
def find_seam(left_img, right_img, left_overlap_line, right_overlap_line):
    height, width=left_img.shape[0], left_img.shape[1]
    left_img_overlap=left_img[:, left_overlap_line:right_overlap_line]
    right_img_overlap=right_img[:, left_overlap_line:right_overlap_line]
    col_length=height
    row_length=right_overlap_line-left_overlap_line

    costs=calc_cost(left_img_overlap, right_img_overlap)

    cost_tab=np.zeros((col_length, row_length))
    path_ids=np.zeros((col_length, row_length), dtype=np.uint32)
    cost_tab[0, :]=costs[0, :]

    for i in range(1, col_length):
        for j in range(row_length):
            best_id=j
            min_cost=cost_tab[i-1, j]
            if j-1>0:
                if cost_tab[i-1, j-1]<min_cost:
                    best_id=j-1
                    min_cost=cost_tab[i-1, j-1]
            if j+1<row_length:
                if cost_tab[i-1, j+1]<min_cost:
                    best_id=j+1
                    min_cost=cost_tab[i-1, j+1]          
            cost_tab[i, j]=min_cost+costs[i, j]  
            path_ids[i, j]=best_id   

    best_path=[]
    best_id = np.argmin(cost_tab[-1])
    for i in range(col_length-1, -1, -1):
        best_path.append([i, best_id+left_overlap_line])
        best_id=path_ids[i, best_id]
        
    best_path.reverse()

    return best_path

# PLOTTING SEAM
def plot_seam(left_img, right_img, seam):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(left_img)
    plt.plot([s[1] for s in seam], [s[0] for s in seam], color='red', linewidth=1)
    plt.title("left image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(right_img)
    plt.plot([s[1] for s in seam], [s[0] for s in seam], color='red', linewidth=1)
    plt.title("right image")
    
    plt.show()

# STITCHING TWO PHOTOS
def stitch_images(left_img, right_img, seam):
    height, width=left_img.shape[0], left_img.shape[1]
    stitched_img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        stitched_img[i, :seam[i][1], :]=left_img[i, :seam[i][1], :]
        stitched_img[i, seam[i][1]:, :]=right_img[i, seam[i][1]:, :]
    return stitched_img

# FINDING BEST HOMOGRAPHY WITH RANSAC
def find_homography_with_ransac(src_pts, dst_pts):
    best_model = None
    max_inliers = 0
    error_t=5

    src_pts=np.array(src_pts)
    dst_pts=np.array(dst_pts)
    for _ in range(1000):
        selected_ids = np.random.choice(len(src_pts), 4, replace=False)
        selected_src_pts = src_pts[selected_ids]
        selected_dst_pts = dst_pts[selected_ids]

        H = find_homography(selected_src_pts, selected_dst_pts)

        src_pts_stacked=np.hstack((src_pts, np.ones((len(src_pts), 1))))
        transformed_pts = np.matmul(H, src_pts_stacked.T).T
        transformed_pts = transformed_pts/transformed_pts[:, 2].reshape(-1, 1)
        transformed_pts = transformed_pts[:, :2]

        diffs = np.linalg.norm(transformed_pts-dst_pts, axis=1)

        inliers = np.where(diffs < error_t)[0]

        if len(inliers) > max_inliers:
            best_model = H
            max_inliers = len(inliers)

    return best_model

# CALCULATING HOMOGRAPHY BASED ON SUPERGLUE MATCHING PAIRS
def calc_homography_based_on_match_pairs(path_matches):
    npz = np.load(path_matches)
    pts1=[]
    pts2=[]
    for i, m in enumerate(npz['matches']):
        if m>=0:
            pts1.append(npz['keypoints0'][i])
            pts2.append(npz['keypoints1'][m])

    pts1=np.array(pts1)
    pts2=np.array(pts2)
    H=find_homography_with_ransac(pts1, pts2)
    return H

# FUNCTIONS FOR DISPLAYING RESULTS OF PARTICULAR TASK

############################
########## TASK 1 ########## 
############################

def task1():
    calibration_images = []
    for i in range(1, 28):
        img = cv2.imread(f"./calibration/img{i}.png")
        calibration_images.append(img)
    
    size=(calibration_images[0].shape[1], calibration_images[0].shape[0])

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # FIRST CALIBRATION METHOD BY USING ALL BOARD
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
    
    # UNDISTORTING CALIBRATION IMAGES TO CHECK CALIRATION EFFECTS
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

    # SECOND CALIBRATION METHOD BY USING ONLY ONE MARKER
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

    # CALCULATING REPROJECTION ERROR
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

    # UNDISTORITING AND SAVING STITCHING PHOTOS BY USING BETTER CAMERA CALIBRATION
    # stitching_images = []
    # for i in range(1, 10):
    #     img = cv2.imread(f"./stitching/img{i}.png")
    #     stitching_images.append(img)

    # for i, img in enumerate(stitching_images):
    #     undistorted_stitching_image = undistort_image(img, cameraMatrix2, distCoeffs2)
    #     comparasion = np.hstack((img, undistorted_stitching_image))
    #     cv2.imshow("Distorted vs Undistorted", comparasion)
    #     # cv2.imshow("Distorted vs Undistorted", undistorted_stitching_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     cv2.imwrite(f"undistorted_stitching/undistorted_img{i+1}.png", undistorted_stitching_image)


############################
########## TASK 2 ########## 
############################

def task2():
    src_img = cv2.imread(f"./undistorted_stitching/undistorted_img1.png")
    # FOUND HOMOGRAPHY IN TASK 4
    H=np.array([[ 7.98740818e-01, -2.10795707e-02,  1.69309897e+02],
        [-4.99802327e-02,  8.80379806e-01,  4.50679763e+01],
        [-1.33520799e-04, -3.90246481e-05,  1.00000000e+00]])

    apply_projective_transformation(src_img, H)
    
############################
########## TASK 3 ########## 
############################
def task3():
    # TESTS
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
def task4():
    src_img = cv2.imread(f"./undistorted_stitching/undistorted_img1.png")
    dst_img = cv2.imread(f"./undistorted_stitching/undistorted_img2.png")

    # MANUALLY ZOOMING AND SELECTING POINTS

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

    # CORRECTING CORDINATES FROM ZOOMING

    def correct_x(x, start, end, width):
        return round(start+(x/width)*(end-start))
    def correct_y(y, start, end, height):
        return round(start+(y/height)*(end-start))

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

    # FOUND HOMOGRAPHY
    H=np.array([[ 7.98740818e-01, -2.10795707e-02,  1.69309897e+02],
        [-4.99802327e-02,  8.80379806e-01,  4.50679763e+01],
        [-1.33520799e-04, -3.90246481e-05,  1.00000000e+00]])

    # FOUND PROJECTIVE TRANSFORMATION
    apply_projective_transformation(src_img, H)


############################
########## TASK 5 ########## 
############################
def task5():

    src_img = cv2.imread(f"./undistorted_stitching/undistorted_img1.png")

    dst_img = cv2.imread(f"./undistorted_stitching/undistorted_img2.png")


    # FOUND H IN TASK 4
    H=np.array([[ 7.98740818e-01, -2.10795707e-02,  1.69309897e+02],
        [-4.99802327e-02,  8.80379806e-01,  4.50679763e+01],
        [-1.33520799e-04, -3.90246481e-05,  1.00000000e+00]])

    src_transformed, min_x, max_x, min_y, max_y=apply_projective_transformation(src_img, H)

    min_xdst, max_xdst, min_ydst, max_ydst=0, 1280, 0, 720
    x_min, x_max, y_min, y_max=min(min_x, min_xdst), max(max_x, max_xdst), min(min_y, min_ydst), max(max_y, max_ydst)
    # print(x_min, x_max, y_min, y_max)

    src_transformed_reshaped=get_new_shape(src_transformed, x_min, x_max, y_min, y_max, min_x, 0)
    # cv2.imshow("src reshaped image", src_transformed_reshaped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    dst_reshaped=get_new_shape(dst_img, x_min, x_max, y_min, y_max, 0, min_y)
    # cv2.imshow("dst reshaped image", dst_reshaped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    best_seam=find_seam(dst_reshaped, src_transformed_reshaped, min_x, max_xdst)
    plot_seam(dst_reshaped, src_transformed_reshaped, best_seam)

    stitched_img12=stitch_images(dst_reshaped, src_transformed_reshaped, best_seam)
    cv2.imshow("stitched image", stitched_img12)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f"task5_stitched12.png", stitched_img12)

############################
########## TASK 6 ########## 
############################

def task6():
    dst_img6 = cv2.imread(f"./undistorted_stitching/undistorted_img8.png")

    src_img6 = cv2.imread(f"./undistorted_stitching/undistorted_img9.png")
    path_matches98 = './matches/undistorted_img9_undistorted_img8_matches.npz'

    H6=calc_homography_based_on_match_pairs(path_matches98)

    src_transformed6, min_x, max_x, min_y, max_y=apply_projective_transformation(src_img6, H6)

    min_xdst, max_xdst, min_ydst, max_ydst=0, 1280, 0, 720
    x_min, x_max, y_min, y_max=min(min_x, min_xdst), max(max_x, max_xdst), min(min_y, min_ydst), max(max_y, max_ydst)
    src_transformed6_reshaped=get_new_shape(src_transformed6, x_min, x_max, y_min, y_max, 0, 0)
    dst_reshaped6=get_new_shape(dst_img6, x_min, x_max, y_min, y_max, -min_x, min_y)

    best_seam98=find_seam(src_transformed6_reshaped, dst_reshaped6, -min_x, max_x)
    plot_seam(src_transformed6_reshaped, dst_reshaped6, best_seam98)
    stitched98=stitch_images(src_transformed6_reshaped, dst_reshaped6, best_seam98)
    cv2.imshow("stitched98", stitched98)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f"task6_stitched98.png", stitched98)

def task7():
    def stitch_5_images(dst_img,src_img1, src_img2, src_img4, src_img5, path_matches13, path_matches23, path_matches43, path_matches53):
        #STITCHING ORDER 3<-2, 32<-1, 4->321, 5->4321

        H13=calc_homography_based_on_match_pairs(path_matches13)
        H23=calc_homography_based_on_match_pairs(path_matches23)
        H43=calc_homography_based_on_match_pairs(path_matches43)
        H53=calc_homography_based_on_match_pairs(path_matches53)
        src_transformed13, min_x13, max_x13, min_y13, max_y13=apply_projective_transformation(src_img1, H13)
        src_transformed23, min_x23, max_x23, min_y23, max_y23=apply_projective_transformation(src_img2, H23)
        src_transformed43, min_x43, max_x43, min_y43, max_y43=apply_projective_transformation(src_img4, H43)
        src_transformed53, min_x53, max_x53, min_y53, max_y53=apply_projective_transformation(src_img5, H53)

        # CORDINATES OF MIDDLE PHOTO
        min_x3, max_x3, min_y3, max_y3=0, 1280, 0, 720
        x_min_all, x_max_all=min(min_x3, min_x13, min_x23, min_x43, min_x53), max(max_x3, max_x13, max_x23, max_x43, max_x53)
        y_min_all, y_max_all=min(min_y3, min_y13, min_y23, min_y43, min_y53), max(max_y3, max_y13, max_y23, max_y43, max_y53)

        # IN OUR CASE WE KNOW THAT x_min_all=min_x53, x_max_all=max_x13, y_min_all<0, min_x53<0 min_x43<0, min_x23>0, min_x13>0
        # RESHAPE TO WHOLE PANORAMA BOARD
        src53_transformed_reshaped=get_new_shape(src_transformed53, x_min_all, x_max_all, y_min_all, y_max_all, 0, y_min_all-min_y53)
        src43_transformed_reshaped=get_new_shape(src_transformed43, x_min_all, x_max_all, y_min_all, y_max_all, -(x_min_all-min_x43), y_min_all-min_y43)
        dst_reshaped=get_new_shape(dst_img, x_min_all, x_max_all, y_min_all, y_max_all, -x_min_all, y_min_all)
        src23_transformed_reshaped=get_new_shape(src_transformed23, x_min_all, x_max_all, y_min_all, y_max_all, -x_min_all+min_x23, y_min_all-min_y23)
        src13_transformed_reshaped=get_new_shape(src_transformed13, x_min_all, x_max_all, y_min_all, y_max_all, -x_min_all+min_x13, y_min_all-min_y13)

        # STITCHING 32
        best_seam32=find_seam(dst_reshaped, src23_transformed_reshaped, -x_min_all+min_x23, -x_min_all+max_x3)
        plot_seam(dst_reshaped, src23_transformed_reshaped, best_seam32)
        stitched32=stitch_images(dst_reshaped, src23_transformed_reshaped, best_seam32)
        # STITCHING 321
        best_seam321=find_seam(stitched32, src13_transformed_reshaped, -x_min_all+min_x13, -x_min_all+max_x23)
        plot_seam(stitched32, src13_transformed_reshaped, best_seam321)
        stitched321=stitch_images(stitched32, src13_transformed_reshaped, best_seam321)
        # STITCHING 4321
        best_seam4321=find_seam(src43_transformed_reshaped, stitched321, -x_min_all, -x_min_all+max_x43)
        plot_seam(src43_transformed_reshaped, stitched321, best_seam4321)
        stitched4321=stitch_images(src43_transformed_reshaped, stitched321, best_seam4321)
        # STITCHING 54321
        best_seam54321=find_seam(src53_transformed_reshaped, stitched4321, -x_min_all+min_x43, -x_min_all+max_x53)
        plot_seam(src53_transformed_reshaped, stitched4321, best_seam54321)
        stitched54321=stitch_images(src53_transformed_reshaped, stitched4321, best_seam54321)

        return stitched54321

    dst_img = cv2.imread(f"./undistorted_stitching/undistorted_img3.png")

    src_img1 = cv2.imread(f"./undistorted_stitching/undistorted_img1.png")
    path_matches13 = './matches/undistorted_img1_undistorted_img3_matches.npz'

    src_img2 = cv2.imread(f"./undistorted_stitching/undistorted_img2.png")
    path_matches23 = './matches/undistorted_img2_undistorted_img3_matches.npz'

    src_img4 = cv2.imread(f"./undistorted_stitching/undistorted_img4.png")
    path_matches43 = './matches/undistorted_img4_undistorted_img3_matches.npz'

    src_img5 = cv2.imread(f"./undistorted_stitching/undistorted_img5.png")
    path_matches53 = './matches/undistorted_img5_undistorted_img3_matches.npz'

    stitched_panorama=stitch_5_images(dst_img,src_img1, src_img2, src_img4, src_img5, path_matches13, path_matches23, path_matches43, path_matches53)
    cv2.imshow("stitched panorama", stitched_panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f"task7_stitched.png", stitched_panorama)

    # for i in range(1000):
    #     selected_four_ids=np.random.choice(n, 4, replace=False)
    #     selected_src_pts=src_pts[selected_four_ids]
    #     selected_dst_pts=dst_pts[selected_four_ids]
    #     H=find_homography(selected_src_pts, selected_dst_pts)
    #     new_dst_pts_arr=np.matmul(H, src_pts_arr.T).T
    #     new_dst_pts_arr[:, 0], new_dst_pts_arr[:, 1]=new_dst_pts_arr[:, 0]/new_dst_pts_arr[:, 2], new_dst_pts_arr[:, 1]/new_dst_pts_arr[:, 2]
    #     diffs=np.sqrt((new_dst_pts_arr[:, 0]-dst_pts_arr[:, 0])**2+(new_dst_pts_arr[:, 1]-dst_pts_arr[:, 1])**2)

    #     inliers_ids=np.where(diffs<error_t)
    #     inliers_num=len(inliers_ids)
    #     if inliers_num>inliers_most:
    #         best_H=H
    #         inliers_most=inliers_num
    #         best_inliers_ids=inliers_ids
    
    # return best_inliers_ids



if __name__ == '__main__':
    #task1()
    #task2()
    #task3()
    #task4()
    #task5()
    #task6()
    task7()
    # src_pts=[[20, 30], [40, 50], [60, 70], [830, 90], [130, 110]]
    # dst_pts=[[20, 30], [40, 50], [60, 70], [80, 90], [100, 110]]
    # print(ransac(src_pts, dst_pts))