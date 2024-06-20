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
from pandas.core.common import flatten
from tqdm import tqdm

def cp2dlparams(calib_param, newK = False):
    i = 0
    if newK == True:
        i = 2
    else:
        i = 0
    return [calib_param[i][0][0], calib_param[i][1][1], calib_param[i][0][2], calib_param[i][1][2]]

def compute_reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    error = np.linalg.norm(image_points - projected_points, axis=1).mean()
    return error

def getAprilTagsInfo(images_folder, calib_param, tag_size, debug_dir=None):
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
    for image in tqdm(files):
        img_name = os.path.basename(image)
        # Initialize as empty list
        tag_info_dict[img_name] = []
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        """  OLD METHOD
        camera_params = (
            camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2])
        tags = at_detector.detect(
            img, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)
        tag_info_dict[img_name] = tags        
        """
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # Detect AprilTags without pose estimation
        tags = at_detector.detect(img)

        # Define the 3D points of the tag's corners in the object coordinate system
        half_size = tag_size / 2
        object_points = np.array([
            [-half_size, -half_size, 0],
            [ half_size, -half_size, 0],
            [ half_size,  half_size, 0],
            [-half_size,  half_size, 0]
        ], dtype=np.float32)

        # Initialize dictionary to store pose information
        #tag_info_dict = {}
        tags_list = []
        # Estimate pose for each detected tag
        for tag in tags:
            tag_tmp = {}

            points_2D = np.array([
                #(tag.center[0], tag.center[1]),
                (tag.corners[3][0], tag.corners[3][1]),
                (tag.corners[2][0], tag.corners[2][1]),
                (tag.corners[1][0], tag.corners[1][1]),
                (tag.corners[0][0], tag.corners[0][1]),
            ], dtype="double")

            points_3D = np.array([
                #(0.0, 0.0, 0.0),
                (-tag_size/2, tag_size/2, 0.0),
                (tag_size/2, tag_size/2, 0.0),
                (tag_size/2, -tag_size/2, 0.0),
                (-tag_size/2, -tag_size/2, 0.0),
            ])
            # Solve PnP using different methods and compare results
            methods = [cv2.SOLVEPNP_IPPE_SQUARE]
            best_rvec, best_tvec = None, None
            best_error = float('inf')
            success = False
            inliers_state = False
            for method in methods:
                success, rvec, tvec = cv2.solvePnP(points_3D, points_2D, calib_param[2], calib_param[1], flags=method)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
                rvec, tvec = cv2.solvePnPRefineVVS(points_3D, points_2D, calib_param[2], calib_param[1], rvec, tvec, criteria)

                if success:
                    error = compute_reprojection_error(points_3D, points_2D, rvec, tvec, calib_param[2], calib_param[1])
                    if error < best_error:
                        best_error = error
                        best_rvec, best_tvec = rvec, tvec
            rvec, tvec = best_rvec, best_tvec
            R = 0
            if success:
                R, _ = cv2.Rodrigues(rvec)
            if success:
                tag_tmp = {
                    'pose_R': R,
                    'pose_t': tvec,
                    'tag_id': tag.tag_id,
                    'center': tag.center,
                    'error': best_error,
                }
                
            tags_list.append(tag_tmp)
        tag_info_dict[img_name] = tags_list
        if len(tags_list) > 0:
            success_dectection += 1

        if debug_dir is not None:
            color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for tag in tags:
                for idx in range(len(tag.corners)):
                    cv2.line(color_img, tuple(
                        tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0), 5)

                cv2.putText(color_img, str(tag.tag_id),
                            org=(tag.corners[0, 0].astype(
                                int)+10, tag.corners[0, 1].astype(int)+10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2,
                            color=(0, 0, 255))

            saveFilename = os.path.join(debug_dir, images_folder.split("/")[-1] + '_debug_' + img_name)
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
    filename_error = filename.replace('.txt', '_errors.txt')
    df = pd.DataFrame(index=list(tag_info_dict.keys()), columns=[
                      'x', 'y', 'z', 'alpha', 'beta', 'gamma', 'center_x', 'center_y', 'error', 'corners'])
    error_types = ['PNP_ITERATIVE', 'PNP_P3P', 'PNP_EPNP', 'PNP_AP3P', 'PNP_IPPE', 'PNP_IPPE_SQUARE', 'PNP_SQPNP']
    
    df_error = pd.DataFrame(index=list(tag_info_dict.keys()), columns=error_types)
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
        center_x = None
        center_y = None
        #ref_center_x = None
        #ref_center_y = None
        return_tag = 0
        ref_taken=False
        #rvec0 = None
        #rvec1 = None
        #rvec2 = None
        error = None
        #errors = None
        #corners = None
        for tag in tags:
            if not tag: #check if dict is empty
                continue
            #if not tag['inlier']:
            #    continue
            if tag['tag_id'] == 0:
                ref_R = tag['pose_R']
                ref_t = tag['pose_t']
                nb_ref_tag +=1
                #ref_center_x = tag['center'][0]
                #ref_center_y = tag['center'][1]
                ref_taken = True
            if tag['tag_id'] == 1:
                goal_R = tag['pose_R']
                goal_t = tag['pose_t'] 
                center_x = tag['center'][0]
                center_y = tag['center'][1]
                #rvec0=tag['rvec'][0]
                #rvec1=tag['rvec'][1]
                #rvec2=tag['rvec'][2]
                error = tag['error']
                #corners = list(flatten(tag['corners']))
                state = 1
                return_tag = 0
                nb_tag1 +=1
                #errors = tag['errors']
            if tag['tag_id'] == 2:
                goal_R = tag['pose_R']
                goal_t = tag['pose_t']
                center_x = tag['center'][0]
                center_y = tag['center'][1]
                #corners = list(flatten(tag['corners']))
                #rvec0=tag['rvec'][0]
                #rvec1=tag['rvec'][1]
                #rvec2=tag['rvec'][2]
                error = tag['error']
                state = 1
                return_tag = 1
                nb_tag2 +=1
                #errors = tag['errors']
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
                df.loc[f, 'center_x'] = center_x#-ref_center_x
                df.loc[f, 'center_y'] = center_y#-ref_center_y
                #df.loc[f, 'rvec0'] = rvec0
                #df.loc[f, 'rvec1'] = rvec1
                #df.loc[f, 'rvec2'] = rvec2
                df.loc[f, 'error'] = error
                #df.loc[f, 'corners'] = corners
                #for error_t, err in zip(error_types, errors):
                #    df_error.loc[f, error_t] = err
    #df_error.to_csv(filename_error, header=False, index=True, mode='w')
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
    #ax.scatter3D(x[0],y[0],z[0], marker = 'o', color = color)
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

def visualize_rvec_df(df, axes, color):
    x = df['rvec0'].to_numpy()
    y = df['rvec1'].to_numpy()
    z = df['rvec2'].to_numpy()
    for i in range(x.shape[0]):
        try:
            axes[0].scatter(i, x[i], marker = "x", color = color)
            axes[1].scatter(i, y[i], marker = "x", color = color)
            axes[2].scatter(i, z[i], marker = "x", color = color)
        except:
            pass
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('rvec0')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('rvec1')
    #axes[2].axis('auto')
    #axes[2].set_ylim(bottom = -0.3, top = 0.3)
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('rvec2')
    #axes[2].set_autoscale_on()
    return axes


def visualize_error_df(df, ax, color):
    error = df['error'].to_numpy()
    for i in range(error.shape[0]):
        try:
            if not np.isnan(error[i]):
                ax.scatter(i, error[i], marker = "x", color = color)
        except:
            pass
    ax.set_xlabel('Step')
    ax.set_ylabel('Error')
    #ax.set_ylim(bottom=-0.1, top=0.1)
    return ax

def visualize_angles_df(df, axes, color):
    alpha = df['alpha'].to_numpy()
    beta = df['beta'].to_numpy()
    gamma = df['gamma'].to_numpy()
    #print(f"gamma max min : {np.max(gamma)}, {np.min(gamma)}")
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
    error = df['error'].to_numpy()
    for i in range(x.shape[0]):
        try:
            axes[0].scatter(i, x[i], marker = "x", color = color)
            axes[1].scatter(i, y[i], marker = "x", color = color)
            axes[2].scatter(i, z[i], marker = "x", color = color)
            if not np.isnan(error[i]):
                axes[3].scatter(i, error[i], marker = "x", color = color)
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
    axes[3].set_xlabel('Step')
    axes[3].set_ylabel('Error')

    #axes[2].set_autoscale_on()
    return axes

def visualize_center_df(df, axes, color):
    center_x = df['center_x'].to_numpy()
    center_y = df['center_y'].to_numpy()
    gamma_max = 0
    gamma_min = 0
    for i in range(center_x.shape[0]):
        try:
            axes[0].scatter(i, center_x[i], marker = "x", color = color)
            axes[1].scatter(i, center_y[i], marker = "x", color = color)
        except:
            pass
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('X')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Y')
    #print(f'gamma_min : {gamma_min}, gamma_ma : {gamma_max}')
    
    return axes

def visualize_corners_df(df, axes, color):
    corner0_x = df[0].to_numpy()
    corner1_x = df[2].to_numpy()
    corner1_y = df[3].to_numpy()
    corner0_y = df[1].to_numpy()
    corner2_x = df[4].to_numpy()
    corner2_y = df[5].to_numpy()
    corner3_x = df[6].to_numpy()
    corner3_y = df[7].to_numpy()
    for i in range(corner0_x.shape[0]):
        try:
            axes[0].scatter(i, corner0_x[i], marker = "x", color = color)
            axes[1].scatter(i, corner0_y[i], marker = "x", color = color)
            axes[2].scatter(i, corner1_x[i], marker = "x", color = color)
            axes[3].scatter(i, corner1_y[i], marker = "x", color = color)
            axes[4].scatter(i, corner2_x[i], marker = "x", color = color)
            axes[5].scatter(i, corner2_y[i], marker = "x", color = color)
            axes[6].scatter(i, corner3_x[i], marker = "x", color = color)
            axes[7].scatter(i, corner3_y[i], marker = "x", color = color)
        except:
            pass

    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('corner 0 x')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('corner 0 y')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('corner 1 x')
    axes[3].set_xlabel('Step')
    axes[3].set_ylabel('corner 1 y')
    axes[4].set_xlabel('Step')
    axes[4].set_ylabel('corner 2 x')
    axes[5].set_xlabel('Step')
    axes[5].set_ylabel('corner 2 y')
    axes[6].set_xlabel('Step')
    axes[6].set_ylabel('corner 3 x')
    axes[7].set_xlabel('Step')
    axes[7].set_ylabel('corner 3 y')

    return axes