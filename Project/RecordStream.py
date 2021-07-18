import time
import cv2


class RecordStreams():
    def __init__(self, save_to_path, chunk_size_in_min=15):
        self.chunk = round(chunk_size_in_min * 60 * 30)
        self.savePath = save_to_path
        self.x = None
        self.y = None

        self.frameCount = 0

        self.chunkNum = 0

    def initializeWriter(self):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        title = 'Stream_'+ str(self.chunkNum) + '.mp4'
        self.vid_writer = cv2.VideoWriter(self.savePath+ '/' + title, fourcc, 30.0, (self.x, self.y))

    def record(self, frame):
        if frame is not None:
            if self.x is None:
                self.y, self.x = frame.shape[:2]

            if self.frameCount % self.chunk == 0:
                self.initializeWriter()
                
            self.vid_writer.write(frame)

            self.frameCount += 1

            if self.frameCount == self.chunk:
                self.vid_writer.release()
                self.chunkNum += 1
        else:
            try:
                self.vid_writer.release()
            except:
                raise StopIteration


input_path = 'D:/PROJECTS/sygmoid/3vision/sample_videos/NLMK/NLMK-Indoorview-01.mp4'

cap = cv2.VideoCapture(input_path)

rec = RecordStreams(save_to_path='D:/PROJECTS/sygmoid/3vision/sample_videos/wayfair', chunk_size_in_min=15)

while True:
    _, frame = cap.read()

    rec.record(frame)