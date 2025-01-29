COMMENTS MY FOR VIRTUAL ENVIRONMENT:
Python version:
Python 3.8.8
Libraries:
mujoco==3.2.3
numpy==1.24.4
opencv-python==4.10.0.84

My code:
I commented code in stub.py file. Here I will describe some important notes about my solution and pseudocode.

COMMENTS FOR TASK 1:
Additinal functions:
find_colors - I USE THIS FUNCTION ALSO IN OTHER TASKS

Main part:
We turn left until the red ball is positioned between 323 and 328 pixels in the x-coordinate of the camera image. in the most time-consuming cases, this process takes up to 40 seconds.
We drive through the red ball until it appears very close,in other words the number of red pixels in the image exceeds a threshold.

COMMENTS FOR TASK 2:
Additional functions: 
find_colors from task1

Main part:
Escaping maze takes around 40 seconds and going to the ball takes another 40 seconds.
Initial movement - turning left while going backward a little bit enables us to avoid getting stuck in the maze corner and find the white/gray pole. Moreover, our car is gray, so we have to crop the image to find a pole.
Then we go: backward, right, forward, left. In the last step, we adjust the position based on the green wall to exit approximately in the center of the maze's exit.
We drive towards red ball like in task 1.

COMMENTS FOR TASK 3:
Additinal functions:
detect_markers - detecting ARUCO markers

Main part:
AS MENTIONED ON SLACK IN THE BEGINNING WE GO A LITTLE BIT FORWARD TO ENSURE TWO MARKERS ARE VISIBLE.
We rotate dash camera to find two markers and after detecting them we rotate back camera.
We teleport our car based on rvec and tvec returned from cv2.solvePnP.
We are close to the ball and adjust our position by moving a little bit backward and turning. Then we turn left to find middle of the ball and we go forward until ball is big enough. 
When we are well positioned we grap the ball.

