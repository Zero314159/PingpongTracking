import numpy as np
import math

class EKF:
    def __init__(self):
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.Q = np.eye(6) * 0.1
        self.R = np.diag((0.1, 0.1, 10, 100, 100, 100))
        self.F = np.eye(6)
        self.H = np.eye(6)
        self.k = 0.01
        self.g = 9.8
        self.last_ts = 0.
    
    def reset(self, x_1):
        self.x = x_1
        self.P = np.eye(6)
        self.F = np.eye(6)
        self.last_ts = 0.

    def predict(self, ts):
        dt = ts - self.last_ts
        v = math.sqrt(self.x[3]**2 + self.x[4]**2 + self.x[5]**2)

        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        self.F[3, 3] = 1 - self.k * v * dt
        self.F[4, 4] = 1 - self.k * v * dt
        self.F[5, 5] = 1 - self.k * v * dt

        self.x[0] += dt * self.x[3]
        self.x[1] += dt * self.x[4]
        self.x[2] += dt * self.x[5]
        self.x[3] -= self.k *  v * self.x[3] * dt
        self.x[4] -= self.k *  v * self.x[4] * dt
        self.x[5] -= self.k *  v * self.x[5] * dt - self.g * dt

        self.P = self.F @ self.P @ self.F.T + self.Q
        # self.last_ts = ts

    def update(self, z_1, ts):
        dt = ts - self.last_ts
        z = np.array([z_1[0], z_1[1], z_1[2], (z_1[0] - self.x[0]) / dt, (z_1[1] - self.x[1]) / dt, (z_1[2] - self.x[2]) / dt])

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P
        self.last_ts = ts


class LSM:
    def __init__(self):
        self.x_list = []
        self.y_list = []
        self.z_list = []
        self.t_list = []
        self.x_coef = [0, 0]
        self.y_coef = [0, 0]
        self.z_coef = [0, 0, 0]

    def reset(self):
        self.x_list = []
        self.y_list = []
        self.z_list = []
        self.t_list = []
        self.x_coef = [0, 0]
        self.y_coef = [0, 0]
        self.z_coef = [0, 0, 0]
    
    def predict(self, ts):
        x = np.zeros(6)
        x[0] = self.x_coef[0] * ts + self.x_coef[1]
        x[1] = self.y_coef[0] * ts + self.y_coef[1]
        x[2] = self.z_coef[0] * ts**2 + self.z_coef[1] * ts + self.z_coef[2]
        x[3] = self.x_coef[0]
        x[4] = self.y_coef[0]
        x[5] = self.z_coef[0] * ts * 2 + self.z_coef[1]
        return x
    
    def update(self, z, ts):
        self.x_list.append(z[0])
        self.y_list.append(z[1])
        self.z_list.append(z[2])
        self.t_list.append(ts)
        if len(self.t_list) > 1:
            self.x_coef = np.polyfit(self.t_list, self.x_list, 1)
            self.y_coef = np.polyfit(self.t_list, self.y_list, 1)
            self.z_coef[1:] = np.polyfit(self.t_list, self.z_list, 1)
        if len(self.t_list) > 2:
            self.z_coef = np.polyfit(self.t_list, self.z_list, 2)
