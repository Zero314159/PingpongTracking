import numpy as np
import cv2
import matplotlib.pyplot as plt
from filter import EKF, LSM
from detector import ball_detector

K = np.array([[865.29032302,   0.,         304.34293431],
            [0.,         866.32284792, 268.04833382],
            [0.,           0.,           1.        ],])
rvecs = np.array([[ 0.03551169],
                [ 0.04409006],
                [-0.02311399],])
tvecs = np.array([[0.06183228],
                [0.11472198],
                [1.96794425],])
R = cv2.Rodrigues(rvecs)[0]

# p_c = np.array([[0.], [0.], [1.]]) * 1.96
# print(np.linalg.inv(R) @ (np.linalg.inv(K) @ p_c - tvecs) * np.array([[1], [-1], [-1]]))

detector = ball_detector()

cap = cv2.VideoCapture("Video_20230917155504053.avi")
cap.set(cv2.CAP_PROP_POS_FRAMES, 60)

use_EKF = False
my_lsm = LSM()
p_w_last = np.zeros(3)
timestamp = 0
EKF_flag = 0
x_o_list = []
x_f_list = []
ts_list = []

while True:
    _, image0 = cap.read()
    while True:
        gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
        keypoints = detector.detect(gray)
        if len(keypoints) == 1:
            kp = keypoints[0]
            cv2.circle(image0, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size / 2), (0, 0, 255), 3)
        elif len(keypoints) > 1:
            maxval = 0
            maxidx = 0
            for idx, kp in enumerate(keypoints):
                cv2.circle(image0, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size / 2), (0, 0, 255), 3)
        
        # print(kp.pt[0], kp.pt[1], kp.size)
        Z = 14.61 * 1.968 / kp.size
        p_c = np.array([[kp.pt[0]], [kp.pt[1]], [1.]]) * Z
        p_w = np.linalg.inv(R) @ (np.linalg.inv(K) @ p_c - tvecs) * np.array([[1], [-1], [-1]])
        p_w = p_w.squeeze()
        # print(p_w)

        if use_EKF:
            if EKF_flag == 0:
                p_w_last = p_w
                EKF_flag = 1
            elif EKF_flag == 1:
                my_ekf = EKF()
                v_1 = (p_w - p_w_last) * 200.0
                p_w_last = p_w
                x_1 = np.hstack((p_w, v_1))
                print(x_1)
                my_ekf.reset(x_1)
                timestamp = 0
                EKF_flag = 2
            else:
                timestamp += 1 / 200.
                my_ekf.predict(timestamp)
                my_ekf.update(p_w, timestamp)
                # print(my_ekf.x)
                # print()
                v_1 = (p_w - p_w_last) * 200.0
                p_w_last = p_w
                x_o = np.hstack((p_w, v_1))
                x_o_list.append(x_o)
                x_f_list.append(my_ekf.x)
                ts_list.append(timestamp)
        else:
            v_o = (p_w - p_w_last) * 200.0
            p_w_last = p_w
            x_o = np.hstack((p_w, v_o))
            x_o_list.append(x_o)
            x_f_list.append(my_lsm.predict(timestamp))
            ts_list.append(timestamp)
            my_lsm.update(x_o, timestamp)
            timestamp += 1 / 200.


        cv2.imshow("demo", image0)
        if cv2.waitKey()==27:
            cap.release()
            cv2.destroyAllWindows()
            
            ax1 = plt.subplot(321)
            ax1.plot(ts_list[1:], np.array(x_o_list)[1:, 0], label="x")
            ax1.plot(ts_list[1:], np.array(x_f_list)[1:, 0], label="x_f")
            ax1.legend()
            ax2 = plt.subplot(323)
            ax2.plot(ts_list[1:], np.array(x_o_list)[1:, 1], label="y")
            ax2.plot(ts_list[1:], np.array(x_f_list)[1:, 1], label="y_f")
            ax2.legend()
            ax3 = plt.subplot(325)
            ax3.plot(ts_list[1:], np.array(x_o_list)[1:, 2], label="z")
            ax3.plot(ts_list[1:], np.array(x_f_list)[1:, 2], label="z_f")
            ax3.legend()
            ax4 = plt.subplot(322)
            ax4.plot(ts_list[1:], np.array(x_o_list)[1:, 3], label="vx")
            ax4.plot(ts_list[1:], np.array(x_f_list)[1:, 3], label="vx_f")
            ax4.legend()
            ax5 = plt.subplot(324)
            ax5.plot(ts_list[1:], np.array(x_o_list)[1:, 4], label="vy")
            ax5.plot(ts_list[1:], np.array(x_f_list)[1:, 4], label="vy_f")
            ax5.legend()
            ax6 = plt.subplot(326)
            ax6.plot(ts_list[1:], np.array(x_o_list)[1:, 5], label="vz")
            ax6.plot(ts_list[1:], np.array(x_f_list)[1:, 5], label="vz_f")
            ax6.legend()
            plt.show()
            exit(0)
        else:
            break