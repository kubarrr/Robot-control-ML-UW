### Python version
python --version:
Python 3.8.8

### According to instruction, full code is in one file stitch.py.

First part of code includes all functions which are used in different tasks:
- apply_projective_transformation - APPLYING PROJECTIVE TRANSFORMATION - USED IN TASKS 2, 4, 5, 6, 7 
- find_homography - FINDING HOMOGRAPHY - USED IN TASKS 3, 4, 5, 6, 7
 - get_new_shape - RESHAPING PHOTO TO PLACE IT ON WHOLE PANORAMA - USED IN TASKS 5, 6, 7
 - calc_cost - CALCULATING COST WITH MODIFICATION TO PENALISE NOT OVERLAPPING REGIONS - USED IN TASKS 5, 6, 7
 - find_seam - FINDING SEAM - USED IN TASKS 5, 6, 7
 - plot_seam - PLOTTING SEAM - CAN BE USED IN TASKS 5, 6, 7
 - stitch_images - STITCHING TWO PHOTOS - USED IN TASKS 5, 6, 7
 - find_homography_with_ransac - FINDING BEST HOMOGRAPHY WITH RANSAC - USED IN TASKS 6, 7
 - calc_homography_based_on_match_pairs - CALCULATING HOMOGRAPHY BASED ON SUPERGLUE MATCHING PAIRS WITH RANSAC- USED IN TASKS 6, 7

Second part of code is divided into 7 functions for particular tasks. We can choose results of which task we want to display while stitch.py execution by uncommenting functions:
if __name__ == '__main__':
    #task1()
    #task2()
    task3()
    #task4()
    task5()
    task6()
    task7()

### Key notes for each task and discussion of task1 results:

#### task1:
Firstly, we calibrate camera with method which uses all available information on the board, (by using full board of markers with known distances between them (function calibrate_camera_all_board())). We undistort calibration photos to see results. Secondly, we calibrate camera by reusing the same image six times (as if there are six ArUcO tags with unknown distances between them (function calibrate_camera_one_marker())). Next we campare reprojection errors of two methods. Results: 
Error for first method is 0.2664872046798019 (known distances between markers)
Error for second method is 0.12271440196217183 (unknown distances)

At the beginning I was suprised by the results. I thought that knowing distances between markers, so in other words having more information about some shapes on the photo will result in more accurate camera calibration. However, smaller reprojection error and in consequence better calibration was achieved by method with unknown distances between markers. I searched for some idea why it happened. I came up with idea that maybe the board is not 100% stiff. It could be a little bit bent and distances could not be exactly preserved. In that case one marker have smaller are and can be more reliable. Moreover having six points of view for markers on different positions can also improve calibration in such situation. Nevertheless I beliece that in perfect, straight world using all available information about image should work better, becasue we can adjust calibration to more visible shapes and reduce error.

Finally, I undistort photos with better calibration method (with smaller reprojection error).

#### task2:
In task 2 we had to implement function which applies projective transformation. In task2() function we execute apply_projective_transformation function on example photo img1, and example homography calculated in task 4 (from igm1 to img2). It is worth to mention that apply_projective_transformation displays origin and transformed image also in next tasks were it is used. Function was implemented in a way described in instruction. It also returns new cordinates of transforemd photo which are then used to find whole panorama plane and to find overlapping regions.

#### task3:
Function find_homography() as I have already mentioned is implemented in first part of stitch.py file. It is used in test_find_homography() function which run 5 tests (for 5 random pair of points) for given homography. Then in loop we make tests for three selected randomly selected homographies.

#### task4:
In this task I implemented zooming and selecting points by mouse clicking functionality. I chose img1 and img2. Selecting points is done manually so main part of code is commented. Based on selected points and found homography there is calculated projective transformation, which is displayed after executing code. It is of course possible to uncomment selecting points part of code and choose differnt matching pairs.

#### task5:
In this task stitching of photos img1->img2 is performed. Homography matrix found in task 4 is used to find projective transformation. Then photos are transformed on one panorama plane (function get_new_shape) and we seam is found (function find_seam()). Plot of seam is commented (function plot_seam()). Then we stitch images (stitch_images). Stitched image is saved to "task_5_stitched.jpg" file.

#### task6:
Procedure of stitching photos img9->img8 is simmilar to this described in task 5. Nevertheless we find homography based on matching pairs from SuperGlue.
!python ./match_pairs.py \
    --input_pairs image_pairs.txt \
    --input_dir /content/ \
    --output_dir /content/output \
    --resize 1280 720 \
    --viz

Function calc_homography_based_on_match_pairs() uses functions find_homography_with_ransac() which calculates best homography based on RANSAC method. Stitched image is saved to "task_6_stitched.jpg" file.

#### task7:
In this task we stitch 5 photos: img1, img2, img3, img4, img5 in order 3<-2, 32<-1, 4->321, 5->4321. This method perform procedure from task 6 four times. Stitched panorama is saved to "task_7_stitched.jpg" file.

#### All parts of code are also explained in stitch.py file (capital letters comments).

