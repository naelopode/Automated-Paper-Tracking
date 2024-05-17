import os
import numpy as np
import glob
import pickle
from vedio2traject_test import *
from preprocessing import *



def getConfig():
    config = {
        'newDetect': True,                  ## frames2taginfo
        'loadInfoDictFromPickle': False,    ## dump frames2taginfo
        'dumpInfoDictPickle': False,        ## load frames2taginfo
        'dumpTxt': True,                    ## dump (x, y, theta)
        'dumpVisualization': False,          ## Draw the trajectory
        'skip' : True,                      ## Skip video to frames
        'log' : True,                       ## Activate logs
        'debug_dir': '/mnt/2To/jupyter_data/PdS_LC/program/automated-paper-tracking/debug'
    }
    return config


if __name__ == "__main__":  
    config = getConfig()

    ## 90 FOV
    #K1 = [926.1512796528766, 0.0, 946.6495715479447, 0.0, 923.4246589255932, 518.5173511065069, 0.0, 0.0, 1.0] #1080p
    #K1 = [1735.8129276308296, 0.0, 2153.028279247778, 0.0, 1707.5843247588116, 1146.9950417793648, 0.0, 0.0, 1.0] #4K
    K1 = [1274.0386514357397, 0.0, 1356.6013306822385, 0.0, 1280.1066490534602, 748.8782982843065, 0.0, 0.0, 1.0] #2.7K
    #K2 = [925.94787353423, 0.0, 973.9713708165266, 0.0, 926.4851632121447, 503.9632841680212, 0.0, 0.0, 1.0] #1080p
    #K2 = [1646.5913473743137, 0.0, 1783.7339500706316, 0.0, 1665.3860630010383, 843.87706850232, 0.0, 0.0, 1.0] #4K
    K2 = [1261.102033160338, 0.0, 1400.1107791391485, 0.0, 1287.2074923618375, 787.0453811970598, 0.0, 0.0, 1.0] #2.7K
    K3 = [1291.3274916479972, 0.0, 1348.0312974451756, 0.0, 1296.5984125995728, 763.0486118239695, 0.0, 0.0, 1.0] #2.7 K
    log = True

    tag_size = 0.034
    tmp_outdir = './working_dir/'
    input_dir = './data/'
    output_dir = './outputs/'

    dirs = [tmp_outdir, input_dir, output_dir]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
    

    ## Grab all in input folder
    # Get the list of files in the folder
    files = os.listdir(input_dir)
    # Create a list to store file paths
    file_paths = []
    vid_prefix = files[0].split('_')[0]
    # Iterate over the file names and construct file paths
    for file_name in files:
        if vid_prefix != file_name.split('_')[0]:
            print('Error, file from different videos in input folder')
            exit()
        file_path = os.path.join(input_dir, file_name)
        file_paths.append(file_path)
    print(f"Files to be treated are {file_paths}") if log else None

    # PREPROCESSING:
    offsets = dump_to_frames(file_paths, skip=config['skip'], log = True)

    for v in file_paths:
        if not os.path.isfile(v):
            print("Check file {} locations. This file does not exists".format(v))
            continue
        v_basename = os.path.basename(v)
        v_basename = v_basename.split('.')[0]
        camera = v_basename.split('_')[1]
        
        print("Working on " + v_basename + "......")

        if camera == "cam1":
            K=K1
        elif camera == "cam2":
            K=K2
        elif camera == "cam3":
            K=K3
        else:
            print('Could not find camera matrix')
            exit()

        video_CM = np.array(K).reshape((3, 3))

        outdir = os.path.join(tmp_outdir, v_basename)
        print(f'outdir is {outdir}')
        if config['newDetect']:
            [tag_info_dict, msg] = getAprilTagsInfo(outdir, video_CM, tag_size, config['debug_dir'])
            print(msg)
        print('done with detecting')

        if config['dumpInfoDictPickle']:
            with open(output_dir + v_basename+'_dict_info.pkl', 'wb') as f:
                pickle.dump(tag_info_dict, f)

        if config['dumpVisualization']:
            figpath = output_dir + v_basename
        else:
            figpath = None

        if config['dumpTxt']:
            dump2txt(tag_info_dict, output_dir + v_basename+'.txt', figpath)

    # Global plots
    fig = plt.figure(figsize=(20,20))
    ax = plt.axes(projection='3d')
    f, axes1 = plt.subplots(3, 1, figsize = (15, 7))
    f2, axes2 = plt.subplots(3, 1, figsize = (15, 7))
    f3, axes3 = plt.subplots(2, 1, figsize = (15, 7))

    missing = []

    files = sorted(glob.glob(os.path.join(output_dir, '*.txt')))

    print(f"File for post-treatment are {files}") if log else None

    for csv, color, offset in zip(files, ['r', 'g', 'b'], offsets):
        df = pd.read_csv(csv, names =['filename','x', 'y', 'z', 'alpha', 'beta', 'gamma', 'center_x', 'center_y'], na_values = missing)
        ax = visualize_xyz_df(df[offset:], ax, color)
        axes1 = visualize_xyz2_df(df[offset:], axes1, color)
        axes2 = visualize_angles_df(df[offset:], axes2, color)
        axes3 = visualize_center_df(df[offset:], axes3, color)
        df[offset:].to_csv(csv.split('.txt')[0]+'_synced.txt', header=False, index=True, mode='w')
    fig.savefig(os.path.join(output_dir,'globalplots_3D.png'))
    f.savefig(os.path.join(output_dir,'globalplots_coord.png'))
    f2.savefig(os.path.join(output_dir,'globalplots_angles.png'))
    f3.savefig(os.path.join(output_dir,'globalplots_2D.png'))
    