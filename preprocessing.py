import cv2
import os
from datetime import datetime
import ffmpeg
from timecode import Timecode
import numpy as np

def time_from_metadata(video, fps, log):
        time_str = ffmpeg.probe(video)["streams"][0]['tags']['timecode']
        print(f'timecode of {video} is {time_str}') if log else None
        tc1 = Timecode(fps, time_str)
        return tc1

def extract_frame_difference(videos, FPS, log=False):
    frames = []
    for video, fps in zip(videos, FPS):
        vid_time = time_from_metadata(video, fps, log)
        frames.append(vid_time.frames)
    print(f'Frames for each video {frames}') if log else None

    order = np.argsort(frames)
    print(f'Sorting order {order}') if log else None
    difference = []
    for i in range(len(frames)):
        difference.append(frames[i]-frames[order[0]])

    print(f'Difference between each video {difference}') if log else None


    return difference


def extract_frames(video_path, skip = False, log = False):
    print(video_path)
    # Create output folder if it doesn't exist
    output_folder = os.path.join('working_dir',video_path.split('/')[-1].split('.')[0])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'Video {video_path} - Has {fps} frames per second') if log else None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Read until video is completed
    current_frame = 0
    while(cap.isOpened() and (not skip)) :
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Save the frame as an image
            frame_path = os.path.join(output_folder, f"frame_{current_frame:05d}.jpg")
            cv2.imwrite(frame_path, frame)

            current_frame += 1
            print(f"Extracting frame {current_frame}/{total_frames}...") if log else None
        else:
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()
    print("Extraction completed.") if log else None
    return fps
    

def dump_to_frames(videos, skip = False, log = False):
    fps = []
    for video in videos:
        fps.append(extract_frames(video, skip, log))
    offset = extract_frame_difference(videos, fps, log)
    return offset