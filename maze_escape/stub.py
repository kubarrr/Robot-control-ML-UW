"""
Stub for homework 2
"""

import time
import random
import numpy as np
import mujoco
from mujoco import viewer


import numpy as np
import cv2
from numpy.typing import NDArray


TASK_ID = 1


world_xml_path = f"car_{TASK_ID}.xml"
model = mujoco.MjModel.from_xml_path(world_xml_path)
renderer = mujoco.Renderer(model, height=480, width=640)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)


def sim_step(
    n_steps: int, /, view=True, rendering_speed = 10, **controls: float
) -> NDArray[np.uint8]:
    """A wrapper around `mujoco.mj_step` to advance the simulation held in
    the `data` and return a photo from the dash camera installed in the car.

    Args:
        n_steps: The number of simulation steps to take.
        view: Whether to render the simulation.
        rendering_speed: The speed of rendering. Higher values speed up the rendering.
        controls: A mapping of control names to their values.
        Note that the control names depend on the XML file.

    Returns:
        A photo from the dash camera at the end of the simulation steps.

    Examples:
        # Advance the simulation by 100 steps.
        sim_step(100)

        # Move the car forward by 0.1 units and advance the simulation by 100 steps.
        sim_step(100, **{"forward": 0.1})

        # Rotate the dash cam by 0.5 radians and advance the simulation by 100 steps.
        sim_step(100, **{"dash cam rotate": 0.5})
    """

    for control_name, value in controls.items():
        data.actuator(control_name).ctrl = value

    for _ in range(n_steps):
        step_start = time.time()
        mujoco.mj_step(model, data)
        if view:
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step / rendering_speed)

    renderer.update_scene(data=data, camera="dash cam")
    img = renderer.render()
    return img



# TODO: add addditional functions/classes for task 1 if needed
# FUNCTION TO FIND COLORS FOR ALL TASKS
def find_colors(img, color):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if color == 'white':
        lower_white = np.array([0, 0, 255]) 
        upper_white = np.array([180, 10, 255])
        lower_gray=np.array([0, 0, 40]) 
        upper_gray=np.array([180, 18, 230])
        mask1 = cv2.inRange(img_hsv, lower_white, upper_white)
        mask2= cv2.inRange(img_hsv, lower_gray, upper_gray)
        mask = mask1 | mask2 

    elif color == 'blue':
        lower_blue = np.array([110, 50, 50]) 
        upper_blue = np.array([130, 255, 255]) 
        mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

    elif color == 'green':
        lower_green = np.array([35, 50, 50]) 
        upper_green = np.array([85, 255, 255]) 
        mask = cv2.inRange(img_hsv, lower_green, upper_green)

    elif color == 'red':
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50]) 
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

    else:
        mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)

    return mask
# /TODO


def task_1():
    steps = random.randint(0, 2000)
    controls = {"forward": 0, "turn": 0.1}
    img = sim_step(steps, view=False, **controls)

    # TODO: Change the lines below.
    # WE TURN LEFT UNTILL RED BALL WILL BE PRESENT BETWEEN 325-330 x COORDINATES IN CAMERA IMAGE
    for _ in range(1000):
        controls = {"forward": 0, "turn": 0.1}
        img = sim_step(10, view=True, **controls)
        red_mask = find_colors(img, color='red')
        red_region = red_mask[:, 323:328]
        if np.any(red_region): 
            print("I detected the ball")
            break

    for _ in range(1000):
        controls = {"forward": 1, "turn": 0}
        img = sim_step(100, view=True, **controls)
        red_mask = np.where(find_colors(img, color='red') > 0, 1, 0)
        red_ball_size = red_mask.sum()
        #print(red_ball_size)

        # WE GO TO THE RED BALL UNTILL IT IS VERY CLOSE, IN OTHER WORDS IT IS BIG ENOUGH
        if red_ball_size > 8000:
            car_position = data.body("car").xpos
            red_ball_position = data.body("target-ball").xpos
            dst= ((car_position[0] - red_ball_position[0])**2 + (car_position[1] - red_ball_position[1])**2) ** 0.5
            print(f"I arrived with distance {dst} to the red ball")
            break

    controls = {"forward": 0, "turn": 0}
    img = sim_step(500, view=True, **controls)
    # /TODO



# TODO: add addditional functions/classes for task 2 if needed
# /TODO

def task_2():
    speed = random.uniform(-0.3, 0.3)
    turn = random.uniform(-0.2, 0.2)
    controls = {"forward": speed, "turn": turn}
    img = sim_step(1000, view=True, **controls)

    # TODO: Change the lines below.
    # For car control, you can use only sim_step function

    # ESCAPING MAZE TAKES AROUND 40S AND GOING TO BALL TAKES ANOTHER 40S
    # INITIAL MOVEMENT - TURNING LEFT WHILE GOING BACKWARD A LITTLE BIT ENABLE US TO NOT GET STUCK IN MAZE CORNER AND FIND WHITE/GRAYPOLE. MOREOVER OUR CAR IS GRAY, SO WE HAVE TO CROP IMAGE.
    for _ in range(10000):
        controls = {"forward": -0.3, "turn": 0.5}
        img = sim_step(10, view=True, **controls)
        img_cropped=img[:340,]
        white_mask = find_colors(img_cropped, color='white')
        white_region = white_mask[:, 50:100]
        if np.any(white_region):
            print("White color detected, I know my position")
            break
    # ADJUSTING POSITION
    controls = {"forward": 0, "turn": -0.08}
    img = sim_step(270, view=True, **controls)
    print("I am going backward")

    # GOING BACKWARD
    controls = {"forward": -0.25, "turn": -0.01}
    img = sim_step(2300, view=True, **controls)
    print("I am ready to turn right")

    # TURNING RIGHT
    controls = {"forward": 0, "turn": -0.32}
    img = sim_step(410, view=True, **controls)
    print("I am ready to go forward")
    
    # GOING FORWARD
    controls = {"forward": 1, "turn": 0}
    img = sim_step(440, view=True, **controls)
    print("I am ready to turn left")

    # TURNING LEFT
    controls = {"forward": 0, "turn": 0.3}
    img = sim_step(250, view=True, **controls)
    print("I am ready to go forward")

    # GOING FORWARD WITH ADJUSTMENT TO AVOID HITTING THE WALL
    turn=0
    for _ in range(160):
        controls = {"forward": 1, "turn": turn}
        img = sim_step(10, view=True, **controls)
        green_mask = find_colors(img, color='green')
        green_region=green_mask[:, -320:-280]
        if np.any(green_region):
            turn=0.3
        else:
            turn=0
    print("I am free:)")    

    # GOING TO THE RED BALL LIKE IN FIRST TASK
    for _ in range(1000):
        controls = {"forward": 0, "turn": 0.1}
        img = sim_step(10, view=True, **controls)
        red_mask = find_colors(img, color='red')
        red_region = red_mask[:, 325:330]
        if np.any(red_region): 
            print("I detected the ball")
            break

    for _ in range(1000):
        controls = {"forward": 1, "turn": 0}
        img = sim_step(100, view=True, **controls)
        red_mask = np.where(find_colors(img, color='red') > 0, 1, 0)
        red_ball_size = red_mask.sum()

        if red_ball_size > 8000:
            car_position = data.body("car").xpos
            red_ball_position = data.body("target-ball").xpos
            dst= ((car_position[0] - red_ball_position[0])**2 + (car_position[1] - red_ball_position[1])**2) ** 0.5
            print(f"I arrived with distance {dst} to the red ball")
            break

    controls = {"forward": 0, "turn": 0}
    img = sim_step(500, view=True, **controls)

    # /TODO



def ball_is_close() -> bool:
    """Checks if the ball is close to the car."""
    ball_pos = data.body("target-ball").xpos
    car_pos = data.body("dash cam").xpos
    print(car_pos, ball_pos)
    return np.linalg.norm(ball_pos - car_pos) < 0.2


def ball_grab() -> bool:
    """Checks if the ball is inside the gripper."""
    print(data.body("target-ball").xpos[2])
    return data.body("target-ball").xpos[2] > 0.1


def teleport_by(x: float, y: float) -> None:
    data.qpos[0] += x
    data.qpos[1] += y
    sim_step(10, **{"dash cam rotate": 0})


def get_dash_camera_intrinsics():
    '''
    Returns the intrinsic matrix and distortion coefficients of the camera.
    '''
    h = 480
    w = 640
    o_x = w / 2
    o_y = h / 2
    fovy = 90
    f = h / (2 * np.tan(fovy * np.pi / 360))
    intrinsic_matrix = np.array([[-f, 0, o_x], [0, f, o_y], [0, 0, 1]])
    distortion_coefficients = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # no distortion

    return intrinsic_matrix, distortion_coefficients


# TODO: add addditional functions/classes for task 3 if needed
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

def detect_markers(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(grayscale_image)
    if ids is not None and len(ids) > 0:
        return corners, ids
    return np.array([]), np.array([])
# /TODO


def task_3():
    start_x = random.uniform(-0.2, 0.2)
    start_y = random.uniform(0, 0.2)
    teleport_by(start_x, start_y)

    # TODO: Get to the ball
    #  - use the dash camera and ArUco markers to precisely locate the car
    #  - move the car to the ball using teleport_by function

    # AS MENTIONED ON SLACK: GOING A LITTLE BIT FORWARD TO ENSURE TWO MARKERS ARE VISIBLE
    controls = {"forward": 1, "turn": 0}
    img = sim_step(100, view=True, **controls)

    # ROTATING DASH CAMERA TO FIND MARKERS
    counts_rot=0 # TO KNOW ANGLE
    detected_markers = {}
    for _ in range(1000):
        img = sim_step(10, view=True, **{"dash cam rotate": -0.05, "forward":0, "lift": 1, "trapdoor close/open": -1})
        counts_rot+=1
        corners, ids = detect_markers(img)
        if len(ids)>1:
            for i, marker_id in enumerate(ids.flatten()):
                detected_markers[marker_id] = corners[i]
            break
    detected_markers=dict(sorted(detected_markers.items()))
    # print(detected_markers)
    # ROTATING DASH CAMERA BACK
    controls = {"forward": 0, "turn": 0, "dash cam rotate": 0.05}
    img=sim_step(10*counts_rot, **controls)

    # DETECTING MARKERS
    intrinsic_matrix, distortion_coefficients=get_dash_camera_intrinsics()
    object_points=np.array([
        [0.05, 0.04, 0.09], 
        [0.05, 0.04, 0.01], 
        [0.05, -0.04, 0.01], 
        [0.05, -0.04, 0.09],
        [-0.04, 0.05, 0.01], 
        [0.04, 0.05, 0.01], 
        [0.04, 0.05, 0.09], 
        [-0.04, 0.05, 0.09]
    ])
    image_points = np.concatenate([marker[0] for marker in detected_markers.values()], axis=0)
    # print(image_points)
    # print(object_points)

    # GETTING POSITION
    _, rvec, tvec = cv2.solvePnP(object_points, image_points, intrinsic_matrix, distortion_coefficients)
    R=cv2.Rodrigues(rvec)[0]
    Rinv=np.linalg.inv(R)
    c=Rinv@tvec
    #print(c)
    teleport_by(c[0][0]+1.22, c[1][0]+2.22)
    # /TODO

    assert ball_is_close()

    # TODO: Grab the ball
    # - the car should be already close to the ball
    # - use the gripper to grab the ball
    # - you can move the car as well if you need to

    # ADJUSTING POSITION
    controls = {"forward": -1, "turn": 0}
    img = sim_step(200, view=True, **controls)
    controls = {"forward": 0, "turn": -1}
    img = sim_step(60, view=True, **controls)

    # FINDING MIDDLE OF THE BALL
    for i in range(1000):
        controls = {"forward": 0, "turn": 0.1}
        img = sim_step(10, view=True, **controls)
        red_mask = np.where(find_colors(img, color='red') > 0, 1, 0)
        red_coords = np.argwhere(red_mask)
        red_mask_middle=red_coords[:, 1].mean()
        if 315<red_mask_middle<325:
            print("I found middle of ball")
            break
    
    # GOING CLOSER TO BALL
    for i in range(1000):
        controls = {"forward": 0.1, "turn": 0}
        img = sim_step(10, view=True, **controls)
        red_mask = np.where(find_colors(img, color='red') > 0, 1, 0)
        # print(red_mask.sum())
        if red_mask.sum()>5400:
            break
    
    # GRABBING THE BALL
    controls = {"forward":0, "lift": -1}
    img = sim_step(200, view=True, **controls)
    
    controls = {"trapdoor close/open": 1}
    img = sim_step(200, view=True, **controls)

    controls = {"lift": 1}
    img = sim_step(2000, view=True, **controls)
    # /TODO

    assert ball_grab()


if __name__ == "__main__":
    print(f"Running TASK_ID {TASK_ID}")
    if TASK_ID == 1:
        task_1()
    elif TASK_ID == 2:
        task_2()
    elif TASK_ID == 3:
        task_3()
    else:
        raise ValueError("Unknown TASK_ID")
