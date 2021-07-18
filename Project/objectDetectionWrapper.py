import argparse
from yolov5base import ObjectDetection
from common.helpers import *
from tracker import *
import time
import os

class ObjectDetectionWrapper():
    def __init__(self, input_path, img_size=416, conf_score=0.6, frame_skip=1):
        opt = self.setArgs(input_path, img_size, conf_score)
        self.frame_skip = frame_skip + 1
        self.object_detection = ObjectDetection(opt, filter_by_class=None)
        self.detector = self.object_detection.predict(frame_skip)

        

    def setArgs(self, input_path, img_size, conf_score):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default='weights/drone.pt', help='model.pt path')
        parser.add_argument('--source', type=str, default=input_path, help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=img_size, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=conf_score, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
        parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        opt = parser.parse_args()      
        return opt 

  
    def displayOutput(self,img,out):
      
        cv2.putText(img,"Aditya",(890,100),cv2.FONT_HERSHEY_PLAIN,5,(0,10,255),6)
        cv2.imshow("Display", cv2.resize(img,(1080,700)))
       
        out.write(img)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            raise StopIteration

    def main(self, isDraw=True, display=True):
        tracker=EuclideanDistTracker()
        if self.isWelding:
                self.weldingTime +=1
    
        fourcc = cv2.VideoWriter_fourcc(*'MPEG') 
        out = cv2.VideoWriter(r'drone.mp4', fourcc, 15, (1920,1080))
     
        
        carcount=[]
        totalcars=[]
        
        while True:
        
            st = time.time()
            originals, all_detections = next(self.detector)
            detlist=[]
        
            for dets in all_detections:    
                objs=dets['class_id']
                x1, y1, x2, y2 = dets['box']
                detlist.append([x1,y1,x2,y2])
            bboxid=tracker.update(detlist)
            for boxid in bboxid:
                a,b,c,d,ids=boxid
                if(ids not in totalcars):
                    totalcars.append(ids)
                cv2.putText(originals,str(ids),(a,b-15),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
                cv2.rectangle(originals, (a,b),(c, d), color.Green, 3)
             
            cv2.putText(originals,str(len(totalcars)),(1700,100),cv2.FONT_HERSHEY_PLAIN,6,(0,10,255),10)
            if display and isDraw:
                self.displayOutput(originals,out)
           
inp_path=r'C:\Users\Aditya\Documents\Aditya\drone\vid.mp4'
det = ObjectDetectionWrapper(input_path=inp_path)
det.main()

