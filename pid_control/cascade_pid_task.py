import mujoco
from mujoco import viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)
viewer.cam.distance = 4.
viewer.cam.lookat = np.array([0, 0, 1])
viewer.cam.elevation = -30.

from drone_simulator import DroneSimulator
from pid import PID

if __name__ == '__main__':
    desired_altitude = 2

    # If you want the simulation to be displayed more slowly, decrease rendering_freq
    # Note that this DOES NOT change the timestep used to approximate the physics of the simulation!
    drone_simulator = DroneSimulator(
        model, data, viewer, desired_altitude = desired_altitude,
        altitude_sensor_freq = 0.01, wind_change_prob = 0.1, rendering_freq = 1
        )

    # TODO: Create necessary PID controllers using PID class
    altitude_pid = PID(
        gain_prop=0.04, gain_int=0.001, gain_der=0.3,
        sensor_period=drone_simulator.altitude_sensor_period,
    )
    acceleration_pid = PID(
        gain_prop=0.005, gain_int=10, gain_der=0.001,
        sensor_period=model.opt.timestep,
    )

    # Increase the number of iterations for a longer simulation
    for i in range(4000):
        # TODO: Use the PID controllers in a cascade designe to control the drone
        current_acceleration = data.sensor("body_linacc").data[2] - 9.8
        if i % 100 == 0:
            desired_acceleration = altitude_pid.output_signal(desired_altitude, drone_simulator.measured_altitudes)
        desired_thrust = acceleration_pid.output_signal(desired_acceleration, [current_acceleration])
        drone_simulator.sim_step(desired_thrust)