import time
import cv2
from queue import Queue
import argparse
import threading
import logging

class Sensor:

    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")
    
class SensorX(Sensor):

    '''Sensor X'''
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self)->int:
        time.sleep(self._delay)
        self._data += 1
        return self._data
    
class SensorCam:

    def __init__(self, name, resolution):
        self._cap = cv2.VideoCapture(name)
        if not self._cap.isOpened():
            raise Exception('Index of camera is wrong.')
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,resolution[1])
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        
    def get(self):
        ret, frame = self._cap.read()
        return [ret, frame]

    def __del__(self):
        self._cap.release()

class WindowImage:

    def __init__(self, delay):
        self._delay = delay

    def show(self,name, img):
        cv2.imshow(name, img)
        
    def __del__(self):
        cv2.destroyAllWindows()

def push (sensor, queue: Queue, frequency):

    while(True):
        if queue.full():
            _ = queue.get_nowait()
        else:
            queue.put_nowait(sensor.get())
            time.sleep(frequency)

if __name__ == '__main__':
    logger = logging.basicConfig(filename = "./log/errors.log", level = logging.ERROR, 
                                 format = '%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='parametrs')
    parser.add_argument('--name', type=int, help='Name of camera', default=0)
    parser.add_argument('--resolution', type=str, help='Resolution', default='640x480')
    parser.add_argument('--frequency', type=float, help='frequency', default=0.001)

    args = parser.parse_args()

    name = args.name
    resolution = args.resolution
    frequency = args.frequency

    resolut = resolution.split('x')
    resol = []
    resol.append(int(resolut[0]))
    resol.append(int(resolut[1]))

    cam = SensorCam(name, resol)
    sen1 = SensorX(1)
    sen2 = SensorX(0.1)
    sen3 = SensorX(0.01)

    qu1 = Queue(2)
    qu2 = Queue(2)
    qu3 = Queue(2)
    quVid = Queue(2)

    thread_video = threading.Thread(target=push, args=(cam, quVid, frequency), daemon=True)
    thread_sen1 = threading.Thread(target=push, args=(sen1, qu1, 0,), daemon=True)
    thread_sen2 = threading.Thread(target=push, args=(sen2, qu2, 0,), daemon=True)
    thread_sen3 = threading.Thread(target=push, args=(sen3, qu3, 0,), daemon=True)

    thread_sen1.start()
    thread_sen2.start()
    thread_sen3.start()
    thread_video.start()

    window = WindowImage(frequency)

 
    frame = None

    Sensor1 = 0
    Sensor2 = 0
    Sensor3 = 0

    while not (cv2.waitKey(1) & 0xFF == ord('q')):
        try:
            if not quVid.empty():
                frame = quVid.get_nowait()
                
                ret = frame[0]
                frame = frame[1]
                if (frame is None):
                    logging.error(Exception('Unable to read the input.'))
                    exit(1)

            if not qu1.empty():
                Sensor1 = qu1.get_nowait()

            if not qu2.empty():
                Sensor2 = qu2.get_nowait()

            if not qu3.empty():
                Sensor3 = qu3.get_nowait()    
        except Exception as error:
            logging.error(error)
        try:      
            if frame is not None:
                frame1 = frame.copy()
                cv2.putText(frame1, 'Sensor1: '+ str(Sensor1), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0,0,0), 1)
                cv2.putText(frame1, 'Sensor2: '+ str(Sensor2), (30, 60), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255,0,0), 1)
                cv2.putText(frame1, 'Sensor3: '+ str(Sensor3), (30, 90), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0,0,255), 1)
                window.show("1",frame1)
        except Exception as out:
            logging.error(f'Camera out {str(out)}')