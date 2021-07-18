from collections import deque
from common.helpers import get_roi, cv2, np
import numpy as np

class KalmanFilter():
    def __init__(self, point, tracker_id, variablesToTrack=None, history=10):
        self.kalman = cv2.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
        # self.kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 0.00003
        self.tracker_id = tracker_id
        self.history = history
        # point['mean'] = np.mean(get_roi(img, new['box']))
        self.past_movements = deque(maxlen=history)
        self.past_movements.append(point)
        while abs(self.kalman.predict()[0] - point['cx']) > 1:
            self.update(point)

        self.obj_not_found = 0
        if variablesToTrack is not None:
            self.variablesToTrack = {key: None for key in variablesToTrack}

    def update(self, obj):
        measurement = np.array([[np.float32(obj['cx'])],[np.float32(obj['cy'])]])
        self.kalman.correct(measurement)

    def predict(self, func):
        x, y = [int(p) for p in self.kalman.predict()[:2]]

        pred = self.past_movements[-1].copy()
        pred['cx'] = x
        pred['cy'] = y
        dx = (pred['box'][2]-pred['box'][0])//2
        dy = (pred['box'][3]-pred['box'][1])//2
        (x1, y1), (x2, y2) = func((x, y), dx, dy, 0)
        pred['box'] = [x1, y1, x2, y2]
        return pred

    def get_track(self):
        return list(self.past_movements)