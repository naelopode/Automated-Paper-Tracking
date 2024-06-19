import os
import numpy as np
import glob
import pickle
from video2traj import *
from preprocessing import *
import ast
import sys

# File naming convention. For ea

def getConfig():
    config = {
        'newDetect': True,                  ## frames2taginfo
        'loadInfoDictFromPickle': False,    ## dump frames2taginfo
        'dumpInfoDictPickle': False,        ## load frames2taginfo
        'dumpTxt': True,                    ## dump (x, y, theta)
        'dumpVisualization': True,         ## Draw the trajectory
        'skip' : False,                     ## Skip video to frames
        'log' : True,                       ## Activate logs
        'debug': True,                      ## Export all frames with tag position
    }
    return config

def convert_string_to_list(value):
    try:
        # First, check if the value is not empty or NaN
        if pd.isna(value) or value == '':
            return np.nan  # or you can return [] for an empty list
        # Use literal_eval to try to parse the list
        result = ast.literal_eval(value)
        if isinstance(result, list):
            return np.array(result)  # or just return result if a list is preferred
        else:
            return result  # In case the evaluated result is not a list
    except:
        # In case of any error during parsing, handle it (return the original or a default value)
        return value

def extract_data(video_folder):
    config = getConfig()
    with open('/mnt/2To/jupyter_data/PdS_LC/program/automated-paper-tracking/cam1_26K.pkl', 'rb') as f:
        calib_param_cam1 = pickle.load(f)
    with open('/mnt/2To/jupyter_data/PdS_LC/program/automated-paper-tracking/cam2_26K.pkl', 'rb') as f:
        calib_param_cam2 = pickle.load(f)
    with open('/mnt/2To/jupyter_data/PdS_LC/program/automated-paper-tracking/cam3_26K.pkl', 'rb') as f:
        calib_param_cam3 = pickle.load(f)

    tag_size = 0.034
    video = video_folder
    input_dir = '/mnt/2To/jupyter_data/PdS_LC/program/all_vids/'+video+'/'
    tmp_outdir = './working_dir/'
    output_dir = input_dir.replace('all_vids', 'all_results')
    # Create debug dir if debug is True
    debug_dir = f'/mnt/2To/jupyter_data/PdS_LC/program/automated-paper-tracking/debug/{video}/' if config['debug'] else None
    os.makedirs(debug_dir, exist_ok=True) if config['debug'] else None
    print(f"DEBUG is {debug_dir}") if config['debug'] else None


    dirs = [tmp_outdir, input_dir, output_dir]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
    

    # Grab all in input folder
    files = os.listdir(input_dir)
    file_paths = []
    vid_prefix = files[0].split('_')[0]
    # Iterate over the file names and construct file paths
    for file_name in files:
        if vid_prefix != file_name.split('_')[0]:
            print('Error, file from different videos in input folder') if config['log'] else None
            exit()
        file_path = os.path.join(input_dir, file_name)
        file_paths.append(file_path)
    print(f"Files to be treated are {file_paths}") if config['log'] else None

    # Extracting frames:
    print(f"dir is {file_path}")
    offsets = dump_to_frames(file_paths, skip=config['skip'], log = config['log'])

    for v in file_paths:
        if not os.path.isfile(v):
            print(f"Check file {v} locations. This file does not exists")
            continue
        v_basename = os.path.basename(v)
        v_basename = v_basename.split('.')[0]
        camera = v_basename.split('_')[1]
        
        print("Working on " + v_basename + "......") if config['log'] else None
        calib_param = None
        # Needs to be adjusted to number of cameras and the naming convention used
        if camera == "cam1":
            calib_param = calib_param_cam1
        elif camera == "cam2":
            calib_param = calib_param_cam2
        elif camera == "cam3":
            calib_param = calib_param_cam3
        else:
            print('Could not find camera matrix, does your files respect the naming convention')
            exit()

        # Prepare output
        outdir = os.path.join(tmp_outdir, v_basename)
        if config['loadInfoDictFromPickle']: # Load results and carry on dataframe building if needed
            with open(output_dir + v_basename+'_dict_info.pkl', 'rb') as f:
                tag_info_dict = pickle.load(f)
        elif config['newDetect']: # Creating a new dict
            print("Starting newDetect")
            [tag_info_dict, msg] = getAprilTagsInfo(outdir, calib_param, tag_size, debug_dir)
            if config['dumpInfoDictPickle']:
                print("Starting dumpInfoDictPickle")
                with open(output_dir + v_basename+'_dict_info.pkl', 'wb') as f:
                    pickle.dump(tag_info_dict, f)
        else:
            print("Either import a pickle a enable newDetect to create one")

        if config['dumpVisualization']:
            print("Starting dumpVisualization")
            figpath = output_dir + v_basename
        else:
            figpath = None

        if config['dumpTxt']:
            print(f'Starting dumpTxt to {output_dir + v_basename}.txt')
            dump2txt(tag_info_dict, output_dir + v_basename+'.txt', figpath)
        if config['newDetect']:
            del tag_info_dict #Delete this large variable.

    print("Finished running, you can visualize the data now")

if __name__ == "__main__":    
    video_folder = sys.argv[1]
    print(f"VIDEO FOLDER IS {video_folder}")
    extract_data(video_folder)