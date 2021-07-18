import cv2
from math import sqrt, ceil, atan2
from common.colors import Colors
import numpy as np
from skimage.measure import compare_ssim

color = Colors()

standard = {'cx': None, 'cy': None, 'box': None, 'class_id': None, 'conf': None}


def load_watermark(path='./logo.png'):
    img = cv2.imread(path)
    return img


def add_logo(logo, frame, alpha=0.75):
    wy, wx = logo.shape[:2]
    y, x = frame.shape[:2]
    # frame = frame[0:y-100]#, 0:x-100]
    img = frame.copy()
    # y, x = frame.shape[:2]

    frame[y-wy-20:y-20, x-wx-20:x-20] = logo

    image_new = cv2.addWeighted(frame, alpha, img, 1 - alpha, 0)

    return image_new

def convert2dictionary(box, confidence, class_id, x_off=0, y_off=0):
    std = standard.copy()
    std['box'] = [int(box[0]+x_off), int(box[1]+y_off), int(box[2]+x_off), int(box[3]+y_off)]
    std['cx'] = int((std['box'][2] + std['box'][0])/2) 
    std['cy'] = int((std['box'][3] + std['box'][1])/2)

    std['class_id'] = class_id
    std['conf'] = confidence.item()
    # self.results.append(std)
    return std

def get_original_img(flag, og, i):
    if flag:
        return og[i].copy()
    else:
        return og

def make_a_box(center, dx, dy, size):
    x1 = max(0, center[0] - dx - size)
    y1 = max(0, center[1] - dy - size)
    x2 = center[0] + dx + size
    y2 = center[1] + dy + size
    return (x1, y1), (x2, y2)

def make_a_box_ppe(center, dx, dy, size):
    x1 = center[0] + dx - size
    y1 = center[1] + dy - size
    x2 = center[0] + dx + size
    y2 = center[1] + dy + size
    return (x1, y1), (x2, y2)


def add_translucent_rect(image, rect, color=color.White, alpha=0.3):
    overlay = image.copy()

    if type(rect) is list:
        for r, col in rect:
            x, y, xw, yh = r  # Rectangle parameters
            cv2.rectangle(overlay, (x, y), (xw, yh), col, -1)  # A filled rectangle
    else:
        x, y, xw, yh = rect  # Rectangle parameters
        cv2.rectangle(overlay, (x, y), (xw, yh), color, -1)  # A filled rectangle
        # cv2.fillPoly(overlay, np.array([polygon]), color)

    image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image_new


def add_translucent_polygon(image, polygon, color, alpha):
    overlay = image.copy()
    # x, y, w, h = 10, 10, 10, 10  # Rectangle parameters
    # cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)  # A filled rectangle
    cv2.fillPoly(overlay, np.array([polygon]), color)

    image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image_new


def get_point_from_roi2original(xr, yr, xo, yo):
    return  xr+xo, yr+yo


def normalize(value, actual_bounds, desired_bounds):
    return desired_bounds[0] + (value - actual_bounds[0]) * (desired_bounds[1] - desired_bounds[0]) / (actual_bounds[1] - actual_bounds[0])


def crop_img(img, x1, y1, x2, y2):
    return img[y1:y2, x1:x2]

def get_roi(img, box):
    roi = crop_img(img, box[0], box[1], box[2], box[3])
    return roi

def draw_circle(img, center, radius, color=[0,100,0], line_thickness=2):
    cv2.circle(img, center, radius, color, line_thickness)

def draw_line(img, p1, p2, color=[100,100,100], thickness=3):
    cv2.line(img, p1, p2, color, thickness)

def draw_text(img, label, location):
    tl=round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    cv2.putText(img, label, location, 0, tl / 3, [225, 255, 255], thickness=max(tl - 1, 1), lineType=cv2.LINE_AA)

def find_min_in_lst_of_dict(lst, key):
    return min(lst, key=lambda x:x[key])

def find_max_in_lst_of_dict(lst, key):
    return max(lst, key=lambda x:x[key])

def get_image_boundary_as_polygon(img_shape, zoom_out):
    return [[0-zoom_out, 0-zoom_out],
            [0-zoom_out, img_shape[0]+zoom_out],
            [img_shape[1]+zoom_out, img_shape[0]+zoom_out],
            [img_shape[1]+zoom_out, 0-zoom_out]]


def checkpoint(h, k, x, y, a, b): 
    return ((x - h)**2 // a**2) + ((y - k)**2 // b**2) 

def calculate_distance(x1, y1, x2, y2):
    return round(sqrt((x1 - x2)**2 + (y1 - y2)**2), 2)

def convert2tuple(item, key1, key2):
    return (item[key1], item[key2])

def find_angle(p1, p2):
    return atan2((p1[1]-p2[1]), (p1[0]-p2[0]))

# Ray tracing
def check_isPoint_inside(x,y,poly):
    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

#PPE detector helpers
def average_color(box):
    blured = cv2.GaussianBlur(box,(25,25),cv2.BORDER_DEFAULT)
    return cv2.mean(blured)

def write_label(img, overlay, labels, loc, shade, text_size=1, text_thickness=2):
    # x1, y1, x2, y2 = loc[0], loc[1]-20, loc[0]+80,loc[1]
    # sub_img = img[y1:y2, x1:x2]
    # white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
    # res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

    # img[y1:y2, x1:x2] = res
    label_width, label_height = cv2.getTextSize(labels[0], cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)[0]

    cv2.rectangle(overlay, (loc[0], loc[1]+label_height*len(labels)), (loc[0]+label_width,loc[1]), shade , -1)
    
    for i, label in enumerate(labels):
        cv2.putText(img, label, (loc[0], loc[1]+16+label_height*i), cv2.FONT_HERSHEY_SIMPLEX , text_size, color.White, text_thickness)
        

def draw_track_path(img, track, color, thickness=3):
    if len(track) > 0:
        X = len(track)
        for i in range(X-1):
            t = ceil((1 + i) / max(1, (X // thickness)))
            draw_line(img, convert2tuple(track[i], 'cx', 'cy'), convert2tuple(track[i+1], 'cx', 'cy'), color, t)
    return img

def find_center_face(x1, x2, nose):
    return x1+(x2 - x1)//2, nose[1]

def predict_face(nose, earL, earR):
    d1 = abs(nose[0] - earL[0])
    d2 = abs(nose[0] - earR[0])
    d = max(d1, d2)
    x1 = nose[0] - d2
    y1 = nose[1] - int(d)
    x2 = nose[0] + d1
    y2 = nose[1] + d
    # [cv2.circle(img, x, 3, [0,0,200], -1) for x in [nose, earL, earR]]
    return x1, y1, x2, y2

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

def calculateCenteroid(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def calculateMinMaxBBOX(arr):
    xMin = np.min(arr[:, 0])
    xMax = np.max(arr[:, 0])
    yMin = np.min(arr[:, 1])
    yMax = np.max(arr[:, 1])
    return np.array([xMin, yMin, xMax, yMax])

def checkGhosts2(img, bbox,  bg, conf):#, mean_w, mean_h):
    # def drw():
    #     cv2.rectangle(bgcopy, tuple(bbox[:2]), tuple(bbox[2:]), color.White, 1)
    #     labels = ["conf: "+str(round(conf*100, 2)), "White: "+str(white), "Black: "+str(black), "percentage: "+str(0 if black ==0 else round(white/TP*100, 2))]
    #     for i, label in enumerate(labels):
    #         cv2.putText(bgcopy, label, (bbox[0], bbox[1]-i*15), cv2.FONT_HERSHEY_PLAIN, 0.85, color.White, 1)
    width, height = bbox[2] - bbox[0], bbox[3]- bbox[1]
    TP= width * height
    roi = get_roi(bg, bbox)
    white = cv2.countNonZero(roi)
    # black= TP - white
    if white/TP > 0.01:# or (conf > 0.2 and white/TP > 0.01):
        # drw()
        # if abs(mean_w - width)/mean_w > 0.2 and abs(mean_h - height)/mean_h > 0.2:
        #     return True
        return False
    return True

def calculate_center_of_bbox(bbox):
    xc = (bbox[0] + bbox[2]) // 2
    yc = (bbox[1] + bbox[3]) // 2
    return xc, yc