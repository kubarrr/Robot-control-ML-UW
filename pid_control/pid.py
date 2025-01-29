# TODO: implement a class for PID controller
class PID:
    def __init__(self, gain_prop, gain_int, gain_der, sensor_period):
        self.gain_prop = gain_prop
        self.gain_der = gain_der
        self.gain_int = gain_int
        self.sensor_period = sensor_period
        # TODO: add aditional variables to store the current state of the controller
        self.prev_err = 0
        self.integral = 0

    # TODO: implement function which computes the output signal
    def output_signal(self, commanded_variable, sensor_readings):
        current_value=sensor_readings[0]
        err=commanded_variable-current_value
        proportional=self.gain_prop * err
        derivative = self.gain_der * (err - self.prev_err) / self.sensor_period
        self.integral += err * self.sensor_period
        integral = self.gain_int * self.integral

        self.prev_err = err

        return proportional + integral + derivative