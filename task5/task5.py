import cv2
from ultralytics import YOLO
import argparse
import string
import multiprocessing
from queue import Queue, Empty
import time
import threading
import numpy as np
class video:
    def __init__(self, num_threads, path, name):
        self.num_threads = num_threads
        self.name = name
        self.paralel(path)
        

    def paralel(self, path):
        self.video_file = cv2.VideoCapture(path)
        video_queue = Queue()
        out_queue = Queue()
        id_of_frame = 0
        self.resulution = None
        while self.video_file.isOpened():
            ret, frame = self.video_file.read()
            try:
                if frame is not None:
                    if self.resulution is None:
                        self.resulution = frame.shape
                    if ret is True:
                        video_queue.put((frame, id_of_frame))
                        id_of_frame += 1
                        print(f"frame {id_of_frame} is read")
                else:
                    break
            except Exception as e:
                print(f'Error in read videofile.{e}')

        n = video_queue.qsize()
        print(n)
        threads = []
        for thread in range(self.num_threads):
            thread = threading.Thread(target=self.thread, args = (video_queue, out_queue,))
            threads.append(thread)
            
        stime = time.time()

        for i in range(self.num_threads):
            threads[i].start()
            print(f"{i} Tread started.")

        for thread in threads:
            thread.join()
            print("Thread is finished")

        frames = [None]*n
       
        try:
            while True:
                try:
                    frame, id = out_queue.get(timeout = 9)
                    frames[id] = frame
                    print(f"Frame {id} is actuality.")
                except Empty:
                    break
        except Exception as e:
            print(f'Frame is broken {e}')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print(len(frames))
        video_writer = cv2.VideoWriter(self.name, fourcc, 30, (self.resulution[1], self.resulution[0]))
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()

        print(time.time() - stime)
                
        
        

    def thread(self, video_queue, out):
        model = YOLO('yolov8n-pose.pt')
        while not video_queue.empty():
            try:
                frame, id = video_queue.get(timeout = 1)
                result = model(frame, device='cpu')[0].plot()
                out.put((result, id))
                video_queue.task_done()
            except Exception as e:
                print(f'Error in YOLO proccesin.{e}')        
    def __del__(self):
        self.video_file.release()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parametrs')
    parser.add_argument('--path', type=str, help='path_to_video', default='no_path')
    parser.add_argument('--threads', type=int, help='count_of_threads', default=1)
    parser.add_argument('--name', type=str, help='name_of_output_file', default='output.mp4')

    arg = parser.parse_args()

    yolca = video(arg.threads, arg.path, arg.name)

