import cv2
from cv2 import imshow
from scipy.spatial.transform import Rotation
from dt_apriltags import Detector
import os
import re
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import math

def getAprilTagsInfo(images_folder, camera_matrix, tag_size, debug_dir=None):
    '''
    Return a dictionary

    tag_info_dict = 
    {
        'frame0.jpg': tags ## list of tags 
        'frame1.jpg': tags ## list of tags
        ...
    }

    and

    msg = Sucess Detect: 302/678
    '''
    at_detector = Detector(searchpath=['/usr/local/lib'],
                           families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)

    success_dectection = 0

    # Get images from folder
    files = sorted(glob.glob(os.path.join(images_folder, '*.jpg')))
    file_num = len(files)
    # Center: k*2, Pose_R: 3*3*k, Pose_t: 3*1*k 
    tag_info_dict = {}
    for image in files:
        img_name = os.path.basename(image)
        # Initialize as empty list
        tag_info_dict[img_name] = []

        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        camera_params = (
            camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2])
        tags = at_detector.detect(
            img, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)
        tag_info_dict[img_name] = tags

        if len(tags) > 0:
            success_dectection += 1

        if debug_dir is not None:
            color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for tag in tags:
                for idx in range(len(tag.corners)):
                    cv2.line(color_img, tuple(
                        tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

                cv2.putText(color_img, str(tag.tag_id),
                            org=(tag.corners[0, 0].astype(
                                int)+10, tag.corners[0, 1].astype(int)+10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.8,
                            color=(0, 0, 255))

            saveFilename = os.path.join(debug_dir, 'debug' + img_name)
            cv2.imwrite(saveFilename, color_img)

    msg = "Success Detect: " + str(success_dectection) + ' / ' + str(file_num)
    return [tag_info_dict, msg]


def visualize_df(df, figpath):
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    z = df['z'].to_numpy()
    alpha = df['alpha'].to_numpy()
    beta = df['beta'].to_numpy()
    gamma = df['gamma'].to_numpy()

    plt.cla()
    ax = plt.figure(figsize=(20,20)).add_subplot(projection='3d')
    ax.set_title('3D Representation')
    ax.scatter3D(x[0],y[0],z[0], marker = 'o', color = 'r')
    for i in range(x.shape[0]):
        try:
            ax.scatter3D(x[i],y[i],z[i], marker = "x", color = 'r')
        except:
            pass
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.axis('equal')
    if figpath is None:
        plt.show()
    else:
        plt.savefig(figpath+ '_3D.png')

    plt.cla()
    f, (ax2, ax3, ax4) = plt.subplots(3, 1, figsize = (15, 7))
    for i in range(x.shape[0]):
        try:
            ax2.scatter(i, alpha[i], marker = "x")
            ax3.scatter(i, beta[i], marker = "x")
            ax4.scatter(i, gamma[i], marker = "x")
        except:
            pass
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Alpha')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Beta')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Gamma')
    if figpath is None:
        plt.show()
    else:
        plt.savefig(figpath+ '_Angles.png')

def dump2txt(tag_info_dict, filename, figpath=None):
    df = pd.DataFrame(index=list(tag_info_dict.keys()), columns=[
                      'x', 'y', 'z', 'alpha', 'beta', 'gamma'])
    print('calling dump2txt')
    nb_ref_tag = 0
    nb_tag1 = 0
    nb_tag2 = 0
    ref_R = 0
    ref_t = 0
    ref_taken = False
    for f, tags in tag_info_dict.items():
        goal_R = 0
        goal_t = 0
        state = 0
        return_tag = 0
        for tag in tags:
            if tag.tag_id == 0:
                ref_R = tag.pose_R
                ref_t = tag.pose_t
                nb_ref_tag +=1
                ref_taken = True
            if tag.tag_id == 1:
                goal_R = tag.pose_R
                goal_t = tag.pose_t 
                state = 1
                return_tag = 0
                nb_tag1 +=1
            if tag.tag_id == 2:
                goal_R = tag.pose_R
                goal_t = tag.pose_t 
                state = 1
                return_tag = 1
                nb_tag2 +=1
            if (state == 1) and (ref_taken):
                rel_pos = goal_t - ref_t
                posB = ref_R.T @ rel_pos
                poseBR = goal_R.T @ ref_R #ref_R.T @ goal_R
                angle = Rotation.from_matrix(poseBR).as_euler(
                    'zyx', degrees=False)
                tag_x = float(posB[0])
                tag_y = float(posB[1])
                tag_z = float(posB[2])
                df.loc[f, 'x'] = tag_x
                df.loc[f, 'y'] = tag_y
                df.loc[f, 'z'] = tag_z
                if return_tag:  
                    angle[0] = angle[0]+ np.pi
                    angle[1] = -angle[1]
                    angle[2] = -np.pi - angle[2]
                for i in range(len(angle)):
                    if angle[i]>np.pi:
                        angle[i] = angle[i]-2*np.pi
                    elif angle[i]<-np.pi:
                        angle[i] = angle[i]+2*np.pi
                df.loc[f, 'alpha'] = angle[0]
                df.loc[f, 'beta'] = angle[1]
                df.loc[f, 'gamma'] = angle[2]

    df.to_csv(filename, header=False, index=True, mode='w')

    print(f"Detected {nb_ref_tag} times the reference tag")
    print(f"Detected {nb_tag1} times the first tag")
    print(f"Detected {nb_tag2} times the second tag")

    # Visualize trajectory
    #visualize_df(df, figpath)


def visualize_xyz_df(df, ax, color):
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    z = df['z'].to_numpy()    
    
    ax.set_title('3D Representation')
    ax.scatter3D(x[0],y[0],z[0], marker = 'o', color = color)
    for i in range(x.shape[0]):
        try:
            ax.scatter3D(x[i],y[i],z[i], marker = "x", color = color)
        except:
            pass
    ax.set_xlabel('X')          
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.axis('equal')
    return ax

    
def visualize_angles_df(df, axes, color):
    alpha = df['alpha'].to_numpy()
    beta = df['beta'].to_numpy()
    gamma = df['gamma'].to_numpy()
    print(f"gamma max min : {np.max(gamma)}, {np.min(gamma)}")
    gamma_max = 0
    gamma_min = 0
    for i in range(alpha.shape[0]):
        try:
            axes[0].scatter(i, alpha[i], marker = "x", color = color)
            axes[1].scatter(i, beta[i], marker = "x", color = color)
            if not np.isnan(gamma[i]):
                if gamma[i]>gamma_max:
                    gamma_max=gamma[i]
                elif gamma[i]<gamma_min:
                    gamma_min=gamma[i]
                axes[2].scatter(i, gamma[i], marker = "x", color = color)
        except:
            pass
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Alpha')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Beta')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Gamma')
    #print(f'gamma_min : {gamma_min}, gamma_ma : {gamma_max}')
    
    return axes
    
def visualize_xyz2_df(df, axes, color):
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    z = df['z'].to_numpy()
    for i in range(x.shape[0]):
        try:
            axes[0].scatter(i, x[i], marker = "x", color = color)
            axes[1].scatter(i, y[i], marker = "x", color = color)
            axes[2].scatter(i, z[i], marker = "x", color = color)
        except:
            pass
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('x')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('y')
    #axes[2].axis('auto')
    #axes[2].set_ylim(bottom = -0.3, top = 0.3)
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('z')
    #axes[2].set_autoscale_on()
    return axes
