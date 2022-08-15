import numpy as np
import pandas as pd
import open3d as o3d
import os
import math
import cv2
import scipy
import io
import argparse
from tqdm import trange,tqdm
import importlib
import zipfile
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

from kitti360scripts.helpers.annotation import Annotation3D, Annotation3DPly, global2local
from kitti360scripts.helpers.labels import id2label, labels, Label
from kitti360scripts.devkits.commons.loadCalibration import readVariable

from utils.data_utils import *
'''
Author: GreatGameDota https://gihub.com/GreatGameDota
Copyright 2022
'''

def project_points(points, num, pose_df):
    pcd_expanded = np.concatenate([points, np.ones((len(points),1))], axis=1)
    
    num = min(pose_df.index.values, key=lambda x:abs(x-num)) # Find closest pose timestamp
    
    pose_t = pose_df.loc[num].values.reshape((3,4))
    pose_expanded = np.concatenate([pose_t, np.array([0,0,0,1]).reshape(1,4)])
    
    car_pose = (np.linalg.inv(pose_expanded) @ pcd_expanded.T).T
    
    return car_pose[:,:3], [0,0,0]

# Modified from: https://github.com/windowsub0406/KITTI_Tutorial
def in_range_points(points, x, y, z, x_range, y_range, z_range):
    """ extract in-range points """
    idxs = np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], y < y_range[1], z > z_range[0], z < z_range[1]))
    return points[idxs], idxs

def crop_img(img,cropx,cropy,top,bottom,left,right):
    startx = cropx-(left//2)
    starty = cropy-(top//2)
    endx = cropx+(right//2)
    endy = cropy+(bottom//2)
    return img[starty:endy,startx:endx]

def project_to_bev(points, colors, bboxes, bbox_colors, scale):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2)

    box_x = bboxes[:, 0]
    box_y = bboxes[:, 1]
    box_z = bboxes[:, 2]
    box_dist = np.sqrt(box_x ** 2 + box_y ** 2)
    
    x_range = (min(np.min(x), np.min(box_x)) - 10, max(np.max(x), np.max(box_x)) + 10)
    y_range = (min(np.min(y), np.min(box_y)) - 10, max(np.max(y), np.max(box_y)) + 10)
    z_range = (min(np.min(z), np.min(box_z)) - 10, max(np.max(z), np.max(box_z)) + 10)
    
    # extract in-range points
    x_lim, idxs_x = in_range_points(x, x, y, z, x_range, y_range, z_range)
    y_lim, idxs_y = in_range_points(y, x, y, z, x_range, y_range, z_range)
    dist_lim, idxs_dist = in_range_points(dist, x, y, z, x_range, y_range, z_range)
    dist_colors = colors[idxs_dist]

    box_x_lim, _ = in_range_points(box_x, box_x, box_y, box_z, x_range, y_range, z_range)
    box_y_lim, _ = in_range_points(box_y, box_x, box_y, box_z, x_range, y_range, z_range)
    box_dist_lim, box_idxs_dist = in_range_points(box_dist, box_x, box_y, box_z, x_range, y_range, z_range)
    box_dist_colors = bbox_colors[box_idxs_dist]

    x_size = int((y_range[1] - y_range[0]))
    y_size = int((x_range[1] - x_range[0]))
    
    # convert 3D lidar coordinates to 2D image coordinates
    x_img = -(y_lim * scale).astype(np.int32)
    y_img = -(x_lim * scale).astype(np.int32)

    box_x_img = -(box_y_lim * scale).astype(np.int32)
    box_y_img = -(box_x_lim * scale).astype(np.int32)

    # shift negative points to positive points (shift minimum value to 0)
    x_img += int(np.trunc(y_range[1] * scale))
    y_img += int(np.trunc(x_range[1] * scale))

    box_x_img += int(np.trunc(y_range[1] * scale))
    box_y_img += int(np.trunc(x_range[1] * scale))
    
    # array to img
    img = np.zeros([y_size * scale + 1, x_size * scale + 1, 3], dtype=np.float32)
    img[y_img,x_img] = dist_colors

    # pad out image
    img = cv2.copyMakeBorder(img, 2000, 2000, 1000, 1000, cv2.BORDER_CONSTANT)

    # crop around center
    center_x = x_img[-1]
    center_y = y_img[-1]
    img = crop_img(img, center_x + 1000, center_y + 2000, 100*scale, 0, 50*scale, 50*scale)
    img = cv2.flip(img, 1)

    # ids = [21] + list(range(45,21,-1)) + list(range(20,9,-1)) + [6,9,8,7]
    ids = [3,2,4,1]
    x_img = img.shape[1]
    y_img = img.shape[0]
    min_x = 0
    min_y = 0
    max_x = img.shape[1]
    max_y = img.shape[0]
    img2 = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.float32)
    for c in ids:
        temp = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.float32)
      
        count = 0
        for i in range(x_img):
            for j in range(y_img):
                color = img[i][j]
                if color[0] != 0 or color[1] != 0 or color[2] != 0:
                    if tuple(color * 255) in color2id2 and color2id2[tuple(color * 255)] == c:
                        temp[i][j] = color
                        count=1
        if count == 0:
            continue

        if c == 4:
            temp = cv2.dilate(temp, np.ones((3,3)))
            temp = cv2.erode(temp, np.ones((3,3)))
        else:
            temp = cv2.dilate(temp, np.ones((9,9)))
            temp = cv2.erode(temp, np.ones((5,5)))

        for i in range(x_img):
            for j in range(y_img):
                color = temp[i][j]
                if color[0] != 0 or color[1] != 0 or color[2] != 0:
                    img2[i][j] = color

        del temp
    img = img2

    # draw bboxes
    img2 = np.zeros([y_size * scale + 1, x_size * scale + 1, 3], dtype=np.float32)
    for i in range(0,len(box_x_img),8):
        ps = []
        for j in range(8):
            ps.append([box_x_img[i+j],box_y_img[i+j]])
        ps=np.array(ps)
        color = box_dist_colors[i]
        
        rect = cv2.minAreaRect(ps)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img2 = cv2.drawContours(img2,[box],0,color,-1)
    
    img2 = cv2.copyMakeBorder(img2, 2000, 2000, 1000, 1000, cv2.BORDER_CONSTANT)
    img2 = crop_img(img2, center_x + 1000, center_y + 2000, 100*scale, 0, 50*scale, 50*scale)
    img2 = cv2.flip(img2, 1)

    for i in range(x_img):
        for j in range(y_img):
            color = img2[i][j]
            if color[0] != 0 or color[1] != 0 or color[2] != 0:
                img[i][j] = color

    # FOV crop
    x, y = img.shape[1] / 2, img.shape[0] - 1
    for length in range(0,1000):
        angle = -90 + x_fov_ang / 2
        endy = int(y + length * math.sin(math.radians(angle)))
        endx = int(x + length * math.cos(math.radians(angle)))

        if endx > -1 and endx < img.shape[1]:
            for i in range(1000):
                img[endy,endx] = [0,0,0]
                endy += 1
                if endy == img.shape[0]:
                    break

        angle = 180 - angle
        endy = int(y + length * math.sin(math.radians(angle)))
        endx = int(x + length * math.cos(math.radians(angle)))

        if endx > -1 and endx < img.shape[1]:
            for i in range(1000):
                img[endy,endx] = [0,0,0]
                endy += 1
                if endy == img.shape[0]:
                    break

    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to generate bev images from accumulated point cloud data.')
    parser.add_argument("--sequence", dest="sequence", help="Sequence to generate images from", default="2013_05_28_drive_0002_sync", type=str, nargs='*')
    parser.add_argument("--scale", dest="scale", help="Resolution of final BEV image", default=10, type=int, nargs='*')
    parser.add_argument("--images", dest="images", help="Save raw BEV images", action='store_true', default=False)
    parser.add_argument("--drive", dest="drive", help="Save result files in Google Drive", action='store_true', default=False)
    parser.add_argument("--data", dest="data", help="Class remapping yaml file to use", default="KITTI-360.yaml", type=str, nargs='*')
    args = parser.parse_args()

    data_path = "data/KITTI-360"
    sequence = args.sequence

    filename = f"{data_path}/calibration/calib_cam_to_velo.txt"
    lastrow = np.array([0,0,0,1]).reshape(1,4)
    cam_to_velo = np.concatenate((np.loadtxt(filename).reshape(3,4), lastrow))

    filename = f"{data_path}/calibration/calib_cam_to_pose.txt"
    fid = open(filename,'r')
    ret = readVariable(fid, "image_00", 3, 4)
    lastrow = np.array([0,0,0,1]).reshape(1,4)
    cam_to_pose = np.concatenate((ret, lastrow))

    cam_df = pd.read_csv(f'{data_path}/data_poses/{sequence}/cam0_to_world.txt', sep=" ", header=None, index_col=0)
    pose_df = pd.read_csv(f'{data_path}/data_poses/{sequence}/poses.txt', sep=" ", header=None, index_col=0)

    intrin = np.array([
                    [552.554261, 0, 682.049453, 0],
                    [0, 552.554261, 238.769549, 0],
                    [0, 0, 1, 0],
    ])
    x_fov_ang = 2 * np.arctan(1408/(2*552.554261))
    x_fov_ang = x_fov_ang * (180/math.pi)

    an = Annotation3D(labelDir=f'{data_path}/data_3d_bboxes', sequence=sequence)
    anPly = Annotation3DPly(labelDir=f'{data_path}/data_3d_semantics', sequence=sequence, isDynamic=True)

    bboxes, bboxes_window, bbox_colors = loadBoundingBoxes(an)
    bboxes2, bboxes_window2, bbox_colors2, times = loadBoundingBoxesDynamic(an)

    windows_unique = np.unique(np.array(bboxes_window), axis=0)
    boxes = []
    allbbox_colors = []
    for window in windows_unique:
        bboxes_ = [bboxes[i] for i in range(len(bboxes)) if bboxes_window[i][0]==window[0]]
        bbox_colors_ = [bbox_colors[i][0] for i in range(len(bbox_colors)) if bboxes_window[i][0]==window[0]]
        
        boxes.append(bboxes_)
        allbbox_colors.append(bbox_colors_)

    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py
    color2id = {}
    for l in labels[6:-1]:
        if l.color in color2id:
            # print(l.color,l.id)
            pass
        color2id[l.color] = l.id
    color2id[(127.5,127.5,127.5)] = 45

    '''
    0 empty (0, 0, 0)
    1 road (7, 9, 15)
    2 sidewalk (8)
    3 ground (6, 10, 22)
    4 building (11, 12, 13, 14, 17, 20, 35, 37, 38, 39, 40, 41)
    5 person (24, 25)
    6 vehicle (26, 27, 28, 29, 30, 31)
    7 bike (32, 33)
    '''
    id_colors = [
                [0, 0, 0],
                [128, 64, 128],
                [244, 35, 232],
                [81, 0, 81],
                [70, 70, 70],
                [255, 0, 0],
                [0, 0, 142],
                [119, 11, 32]
    ]

    color2id2 = {}
    count_ = 0
    for c in id_colors:
        color2id2[tuple(c)] = count_
        count_ += 1

    root = f"{data_path}/data_accumulated/"
    paths = os.listdir(root)
    paths.sort()

    output_dir = f"{data_path}/bev_images"
    if os.path.isdir(output_dir):
        os.system(f'rm -r {output_dir}')
    os.mkdir(output_dir)

    output_dir2 = f"{data_path}/seg_maps"
    if os.path.isdir(output_dir2):
        os.system(f'rm -r {output_dir2}')
    os.mkdir(output_dir2)

    if args.drive:
        os.system("rm -r '../drive/My Drive/seg_maps/'")
        os.system("mkdir '../drive/My Drive/seg_maps/'")

    # OUT_TRAIN = '../drive/My Drive/image.zip'
    # with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out:
    for p in tqdm(paths):
        filename = f'{root}{p}'
        pcd = o3d.io.read_point_cloud(filename)
        num = int(filename.split("_")[-1].split(".")[0])
        # num = int(filename.split("_")[-2])

        # Point Cloud
        points = np.asarray(pcd.points)
        
        colors = np.asarray(pcd.colors)
        colors = remapColors((colors * 255).astype(np.uint8), f"data/{args.data}", dtype="points") / 255.

        points, center = project_points(points, num, pose_df)
        points = np.concatenate([points,[center]], axis=0)
        colors = np.concatenate([colors,[[0,0,0]]],axis=0)
        
        # Bbox
        ids = [5,6,7]
        bbox_points = []
        bbox_pts_color = []
        for idx in range(len(boxes)):
            boxs = boxes[idx]
            allbbox_colors[idx] = remapColors(np.array(allbbox_colors[idx]) * 255, f"data/{args.data}", dtype="boxes")
            for j,box in enumerate(boxs):
                colors_ = allbbox_colors[idx][j]
                bounding = o3d.geometry.TriangleMesh.get_oriented_bounding_box(box)
                
                if tuple(colors_) in color2id2 and color2id2[tuple(colors_)] in ids:
                    for k in range(8):
                        point = np.asarray(bounding.get_box_points())[k]
                        bbox_points.append(point)
                        bbox_pts_color.append(colors_)
                        
        # Dynamic Bbox
        for idx in range(len(bboxes2)):
            boxs = bboxes2[idx]
            time = min(times[idx], key=lambda x:abs(x-num)) # Find closest box
            j = np.where(np.array(times[idx])==time)[0][0]
            
            box = bboxes2[idx][j]
            colors_ = bbox_colors2[idx][j][0] * 255
            colors_ = remapColors(np.array(colors_)[None], f"data/{args.data}", dtype="boxes")
            timestamp = times[idx][j]
            bounding = o3d.geometry.TriangleMesh.get_oriented_bounding_box(box)

            if abs(timestamp - num) < 11 and tuple(colors_) in color2id2 and color2id2[tuple(colors_)] in ids:
                for k in range(8):
                    point = np.asarray(bounding.get_box_points())[k]
                    bbox_points.append(point)
                    bbox_pts_color.append(colors_)

        bbox_points = np.array(bbox_points)
        bbox_pts_color = np.array(bbox_pts_color) / 255.
        
        bbox_points_proj, _ = project_points(bbox_points, num, pose_df)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(np.concatenate([points,bbox_points_proj], axis=0))
        # pcd.colors = o3d.utility.Vector3dVector(np.concatenate([colors,bbox_pts_color], axis=0))

        # Project
        top_image = project_to_bev(points, colors, bbox_points_proj, bbox_pts_color, scale=args.scale)

        # Save image
        image = top_image * 255
        image = image.astype(np.uint8)

        seg_map = np.zeros([image.shape[0], image.shape[1], 1], dtype=np.int32)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                seg_map[i][j] = color2id2[tuple(image[i][j])]

        name = sequence + "_" + filename.split("_")[-2] + "_" + filename.split("_")[-1].split(".")[0]
        if args.drive:
            np.save(f'../drive/My Drive/seg_maps/{name}.npy', seg_map)
        np.save(f'{output_dir2}/{name}.npy', seg_map)

        if args.images:
            img = Image.fromarray(image)
            img.save(f'{output_dir}/{name}.png', "PNG")

        # image = scipy.misc.toimage(image, high=np.max(image), low=np.min(image))
        # file_object = io.BytesIO()
        # image.save(file_object, "PNG")
        # image.close()
        # img_out.writestr("data/KITTI-360/bev_images/" + name + '.png', file_object.getvalue())

        # break

        del top_image
        del image