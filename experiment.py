import requests
from time import sleep
import pyfirmata
from tkinter import *
from time import sleep
import os 

    

retry = True
test = False
ARDUINO_PORT = '/dev/ttyACM0'

cam1 = '172.20.133.51'
cam2 = '172.20.137.51'
cam3 = '172.26.141.51'
cams = [cam1, cam2, cam3]

ANGLE_REST = 50
ANGLE_GRAB = 190


def move_servo(angle):
    pin9.write(angle)

    
def drop():
    global pin9
    board=pyfirmata.Arduino(ARDUINO_PORT)
    #iter8 = pyfirmata.util.Iterator(board)
    #iter8.start()
    pin9 = board.get_pin('d:9:s')

    for i in range(ANGLE_GRAB, ANGLE_REST, -1):
        print(f'angle {i}')
        move_servo(i)
        sleep(0.0015)

def move_range():
    global pin9
    board=pyfirmata.Arduino(ARDUINO_PORT)
    #iter8 = pyfirmata.util.Iterator(board)
    #iter8.start()
    pin9 = board.get_pin('d:9:s')


    for i in range(10, 170):
        print(f"angle {i}")
        move_servo(i)
        sleep(0.01)
    for i in range(170, 10, -1):
        print(f'angle {i}')
        move_servo(i)
        sleep(0.01)

def grab():
    global pin9
    board=pyfirmata.Arduino(ARDUINO_PORT)
    #iter8 = pyfirmata.util.Iterator(board)
    #iter8.start()
    pin9 = board.get_pin('d:9:s')


    for i in range(ANGLE_REST, ANGLE_GRAB):
        print(f"angle {i}")
        move_servo(i)
        sleep(0.0015)
    sleep(4)

def cams_enable(cams):
    for i in range(len(cams)):
        print(f'currently at {i}')
        cam = cams[i]
        try:
            print(f"turning on {cam}")
            r = requests.get(f"http://{cam}:8080/gopro/camera/control/wired_usb?p=1", timeout=2)
            print(r.json())
        except requests.exceptions.Timeout:
            i = i-1
            print('The request timed out')
    return True

def cams_disable(cams):
    for i in range(len(cams)):
        print(f'currently at {i}')
        cam = cams[i]
        try:
            print(f"turning off {cam}")
            r = requests.get(f"http://{cam}:8080/gopro/camera/control/wired_usb?p=0", timeout=2)
            print(r.json())
        except requests.exceptions.Timeout:
            i = i-1
            print('The request timed out')
    return True

def cams_start_recording(cams):
    for i in range(len(cams)):
        print(f'currently at {i}')
        cam = cams[i]
        try:
            print(f"recording on cam {cam}")
            r = requests.get(f"http://{cam}:8080/gopro/camera/shutter/start", timeout=2)
            print(r.json())
        except requests.exceptions.Timeout:
            i=i-1
            print('The request timed out')
    return True

def cams_stop_recording(cams):
    for i in range(len(cams)):
        print(f'currently at {i}')
        cam = cams[i]
        try:
            print(f"stop recording on cam {cam}")
            r = requests.get(f"http://{cam}:8080/gopro/camera/shutter/stop", timeout=2)
            print(r.json())
        except requests.exceptions.Timeout:
            i = i-1
            print('The request timed out')
    return True

def get_files(cams):
    files = ['','','']
    times = ['', '', '']
    for i in range(len(cams)):
        cam = cams[i]
        try:
            print(f"reaching {cam}")
            r = requests.get(f"http://{cam}:8080/gopro/media/list", timeout = 2)
            #print(r.json())
            data = r.json()   
            list = []
            cre = []
            for el in data['media']:
                for al in el['fs']:
                    list.append(al['n'])
                    cre.append(al['cre'])
            list.sort()
            cre.sort()
            #print(cre)
            #print(list[-1])
            files[i] = list[-1]
            times[i] = cre[-1]
        except requests.exceptions.Timeout:
            i = i-1
            print('The request timed out')
    print(times)
    return files

def download_files(cams, files, X):
    if not os.path.exists(f'/home/nael/recordings/vid{X}'):
        os.mkdir(f'/home/nael/recordings/vid{X}')
    else:
        print('cauting rewriting on file')
    for i in range(len(cams)):
        cam = cams[i]
        try:
            print(f"reaching {cam}")
            filename = files[i]
            print(filename)
            url = f"http://{cam}:8080/videos/DCIM/100GOPRO/{filename}"
            r = requests.get(url,stream=True, timeout=20)
            new_file = f'/home/nael/recordings/vid{X}/vid{X}_cam{i+1}.MP4'
            with open(new_file, "wb") as out_file:
                print(f'writing content to {new_file}')
                out_file.write(r.content)    
        except requests.exceptions.Timeout:
            i = i-1
            print('The request timed out')
    return True

def record_download(cams, X):
    cams_disable(cams)
    sleep(1)
    cams_enable(cams)
    sleep(1)
    cams_start_recording(cams)
    drop()
    sleep(3)
    cams_stop_recording(cams)
    sleep(1)
    files = get_files(cams)
    sleep(1)
    download_files(cams, files, X)

    cams_disable(cams)

def redownload(cams, X):
    files = get_files(cams)
    sleep(1)
    download_files(cams, files, X)

def set_time(cams):
    cams_disable(cams)


cams_disable(cams)

if test:
    move_range()
    move_range()
    move_range()
    sleep(1)
    grab()
    sleep(1)
    drop()
    exit()
if retry:
    X = 100
    path = f'/home/nael/recordings/vid{X}'
    while os.path.exists(path):
        print('file exist, to next path')
        X = X+1
        path = f'/home/nael/recordings/vid{X}'
    X=X-1

    cams_enable(cams)
    redownload(cams, X)
    cams_disable(cams)
else:
    X = 100
    path = f'/home/nael/recordings/vid{X}'

    while os.path.exists(path):
        print('file exist, to next path')
        X = X+1
        path = f'/home/nael/recordings/vid{X}'

    grab()
    sleep(1)
    record_download(cams, X)

