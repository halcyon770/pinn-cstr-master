# create by qilin
# create in 20220105
# pid controller
import numpy as np


class PID:
    def __init__(self, y_set, y0, sim_step, error_recorder):
        self.sim_step = np.array(sim_step)
        self.Kp = [3, 3]
        self.Ki = [1, 1]
        self.Kd = [1, 1]
        self.sample_time = 0.00
        self.last_error = 0.00
        self.y_set = np.array(y_set)
        self.y0 = np.array(y0)
        self.PTerm = 0
        self.DTerm = 0
        self.ITerm = 0
        self.output = 0
        self.error_recorder = np.array(error_recorder)

    def controller(self):
        error = self.y_set - self.y0
        self.PTerm = self.Kp * error  # 比例
        self.ITerm = self.error_recorder  # 积分
        self.DTerm = 0.0
        self.output = self.PTerm + np.multiply(self.Ki, self.ITerm) + np.multiply(self.Kd, self.DTerm)
        delta_error = error - self.last_error
        return self.output

