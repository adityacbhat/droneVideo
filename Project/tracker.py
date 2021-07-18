from common.helpers import *
from common.kalman_filter import KalmanFilter
from copy import deepcopy
from collections import deque

def convert(box):
    return {'x1':box[0], 'y1':box[1], 'x2':box[2], 'y2':box[3]}

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    bb1 = convert(bb1)
    bb2 = convert(bb2)
    # assert bb1['x1'] < bb1['x2']
    # assert bb1['y1'] < bb1['y2']
    # assert bb2['x1'] < bb2['x2']
    # assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    # assert iou >= 0.0
    # assert iou <= 1.0
    return iou #inverting




class Track():
    def __init__(self, frame_skip=5, variablesToTrack=None, TrackHistory=10):
        self.tracks = []
        self.uid = 0
        self.frame_skip = frame_skip
        self.variablesToTrack = variablesToTrack
        self.TrackHistory = TrackHistory

        self.lostNfound = []

        self.imgArr = deque(maxlen=2)

    def update(self, detections, poly, img=None, cost_thresh=0.1, uid_mod=10000):
        self.imgArr.append(gray(img))
        detections = [dets for dets in detections if dets['class_id'] == 'person']
        new_detections = deepcopy(detections)
        # loop for updating existing objects from yolo detections
        for kf in reversed(self.tracks):
            last = kf.past_movements[-1]
            max_prob = self.track_object(last, new_detections, 'new')
            try:
                score = compare_ssim(get_roi(self.imgArr[0], last['box']), get_roi(self.imgArr[-1], last['box']), full=True)[0]
            except:
                score = 0
            if max_prob['cost'] > cost_thresh:
                kf.update(max_prob['point'])
                kf.past_movements.append(max_prob['point'])
                kf.obj_not_found = 0 
                new_detections.remove(max_prob['point'])
            elif score > 0.95:
                kf.update(max_prob['point'])
                kf.past_movements.append(max_prob['point'])
                kf.obj_not_found = 0 
            else:
                pred = kf.predict(make_a_box)
                kf.update(pred)
                kf.past_movements.append(pred)
                kf.obj_not_found += 1
        # loop for updating objects from past movements if not create new objects and track lost and found
        for new in new_detections:
            max_prob = self.track_object(new, self.tracks, None)
            if max_prob['cost'] > cost_thresh:
                max_prob['point'].update(new)
                max_prob['point'].obj_not_found = 0
            else:
                found = self.track_object(new, self.lostNfound, None)
                if found['cost'] > cost_thresh:
                    found['point'].update(new)
                    found['point'].obj_not_found = 0
                    self.tracks.append(found['point'])
                    self.lostNfound.remove(found['point'])
                else:
                    self.uid = ((self.uid + 1)%uid_mod) + 1
                    # self.tracks.append(Person(new, self.uid))
                    self.tracks.append(KalmanFilter(new, self.uid, self.variablesToTrack, self.TrackHistory))
        # memory check for lost and found
        for lnf in reversed(self.lostNfound):
            # x,y,w,h = lnf.past_movements[-1]['box']
            # cv2.rectangle(img, (x,y),(w, h), [255,50,50], 1)
            # cv2.putText(img, str(lnf.obj_not_found), (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, [100,200,200], 1)

            lnf.obj_not_found += 1
            if lnf.obj_not_found > 100:
                self.lostNfound.remove(lnf)
        self.update_object_existnaces(poly)

    def track_object(self, obj, arr, track):
        points = []
        if len(arr) == 0:
            max_prob = {'point': obj, 'cost': 0}
            return max_prob
        for point in arr:
            if point is None:
                # print("array------------",arr)
                return {'point': obj, 'cost': 0}
            # print("array", arr)
            # d = {'point': point, 'distance': calculate_distance(last['cx'], last['cy'], point['cx'], point['cy'])}
            if track == 'new': 
                # d = {'point': point, 'cost': calculate_distance(obj['cx'], obj['cy'], point['cx'], point['cy'])}
                d = {'point': point, 'cost': get_iou(obj['box'], point['box'])}

            else:
                p = point.past_movements[-1]
                # d = {'point': point, 'cost': calculate_distance(obj['cx'], obj['cy'], p['cx'], p['cy'])}
                d = {'point': point, 'cost': get_iou(obj['box'], p['box'])}

            points.append(d)
        max_prob = find_max_in_lst_of_dict(points, key='cost')
        # max_prob = find_max_in_lst_of_dict(points, key='iou')
        return max_prob
        

    def update_object_existnaces(self, poly):
        for kf_obj in reversed(self.tracks):
            # if not check_isPoint_inside(kf_obj.past_movements[-1]['cx'], kf_obj.past_movements[-1]['cy'], poly):
            #     self.tracks.remove(kf_obj)
            if kf_obj.obj_not_found > self.frame_skip-1:
                self.lostNfound.append(kf_obj)
                self.tracks.remove(kf_obj)
                

    
        
