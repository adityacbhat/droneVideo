
from utils.datasets import *
from utils.utils import *
from common.helpers import convert2dictionary, get_original_img
import time
# from analytics.load_model import load_model

class ObjectDetection():
    def __init__(self, opt, filter_by_class=None):
        with torch.no_grad():
            self.opt = opt
            opt.img_size = check_img_size(opt.img_size)

            source, weights, imgsz = opt.source, opt.weights, opt.img_size
            self.webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

            # Initialize
            self.device = torch_utils.select_device(opt.device)

            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
            # Load model
            google_utils.attempt_download(weights)

            self.model = torch.load(weights, map_location=self.device)['model'].float()  # load to FP32
            # self.model = load_model(weights).float()
            # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
            # model.fuse()
            self.model.to(self.device).eval()
            if self.half:
                self.model.half()  # to FP16

            # Second-stage classifier
            # classify = False
            # if classify:
            #     modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            #     modelc.to(device).eval()
            if self.webcam:
                torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
                self.dataset = LoadStreams(source, img_size=imgsz)
            else:
                self.dataset = LoadImages(source, img_size=imgsz)

            # Get names and colors
            self.names = self.model.names if hasattr(self.model, 'names') else self.model.modules.names
            try:
                self.class_filter = None if filter_by_class is None else [self.names.index(x) for x in filter_by_class.split(',')]
            except:
                print("\"{}\" not in list. Please input proper names!!\nClass Names: {}".format(filter_by_class, self.names))
            # self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
            # Run inference
            img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
            _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
            self.count = 0

    def predict(self, frame_skip=0):
        for path, img, im0s, vid_cap in self.dataset:
            self.count += 1
            if self.count % frame_skip == 0:
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                
                # Inference
                # t1 = torch_utils.time_synchronized()
                pred = self.model(img, augment=self.opt.augment)[0]
                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,
                                        fast=True, classes=self.class_filter, agnostic=self.opt.agnostic_nms)
                # t2 = torch_utils.time_synchronized()

                # Apply Classifier
                # if classify:
                #     pred = apply_classifier(pred, modelc, img, im0s)

                # Process detections
                all_detections = []
                for i, det in enumerate(pred):  # detections per image
                    im0 = get_original_img(self.webcam, im0s, i)

                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Write results
                        for *xyxy, conf, cls in det:
                            std_dict = convert2dictionary(xyxy, conf, self.names[int(cls)])

                            all_detections.append(std_dict)
            else:
                im0 = get_original_img(self.webcam, im0s, 0)
                all_detections = []

            yield im0, all_detections

