import numpy as np
import open3d as o3d
import os
import argparse
from tqdm import trange
import importlib

from recoverKITTI360label.accumulation import PointAccumulation
from utils.data_utils import assignColorLocal
'''
Author: GreatGameDota https://gihub.com/GreatGameDota
Copyright 2022
'''

def recoverLabels(root_dir, sequence, output_dir, first_frame, last_frame, verbose=True, source=1, travel_padding=1, min_dist_dense=0.02, downSampleEvery=-1):
    all_spcds = os.listdir(os.path.join(os.path.join(os.path.join(root_dir,"data_3d_semantics/train"),sequence),"static"))
    all_spcds.sort()

    PA = PointAccumulation(root_dir, output_dir, sequence, first_frame, last_frame, travel_padding, source, min_dist_dense, verbose_=False, compute_labels=True)

    range_ = [int(f.split("_")[0]) for f in all_spcds]
    for j,r in enumerate(range_):
      if r > first_frame:
        i = j - 1
        break
    
    spcd = all_spcds[i]
    if i == 0:
        spcd_prev = None 
    else:
        spcd_prev = all_spcds[i-1]
    if i == len(all_spcds)-1:
        spcd_next = None 
    else:
        spcd_next = all_spcds[i+1]

    success = True
    if not PA.loadTransformation():
        print('Error: Unable to load the calibrations!')
        success = False

    if not PA.getInterestedWindow():
        print('Error: Invalid window of interest!')
        success = False

    if success:
        if verbose:
            print("Loaded " + str(len(PA.Tr_pose_world)) + " poses")

        PA.loadTimestamps()

        if verbose:
            print("Loaded " + str(len(PA.veloTimestamps)) + " velo timestamps")

        PA.addVelodynePoints()
        PA.getPointsInRange()
        PA.recoverLabel(spcd,spcd_prev,spcd_next,0.5)
    else:
      return -1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(PA.Md)

    # Assign Color
    localIds = PA.labels
    ptsColor = assignColorLocal(localIds, "semantic")
    pcd.colors = o3d.utility.Vector3dVector(ptsColor)

    color = np.asarray(pcd.colors)
    mask = np.logical_and(np.mean(color,1)==0, np.std(color,1)==0)
    mask = 1-mask
    pcd = pcd.select_by_index(np.where(mask)[0])

    if downSampleEvery>1:
        pcd = pcd.uniform_down_sample(downSampleEvery)

    _ = o3d.io.write_point_cloud(f'{output_dir}/2013_05_28_drive_0000_sync_{first_frame:06d}_{last_frame:06d}.ply', pcd)
    # os.system("mkdir data/KITTI-360/outputs/labels/")
    # np.save(f'{output_dir}/labels/2013_05_28_drive_0000_sync_{first_frame:06d}_{last_frame:06d}_labels.npy', PA.labels[:,None])
    return pcd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to accumulate all labeled static point cloud points and save them to file.')
    parser.add_argument("sequence", help="Sequence to accumulate for", default="2013_05_28_drive_0002_sync", type=str, nargs='*')
    parser.add_argument("frames", help="Amount of frames to accumulate over", default=10, type=int, nargs='*')
    args = parser.parse_args()

    root_dir = "data/KITTI-360"
    sequence = args.sequence
    output_dir = "data/KITTI-360/outputs"
    if os.path.isdir("data/KITTI-360/outputs/"):
        os.system("rm -r data/KITTI-360/outputs/")
    os.mkdir("data/KITTI-360/outputs/")
    frames = args.frames

    all_spcds = os.listdir(os.path.join(os.path.join(os.path.join(root_dir,"data_3d_semantics/train"),sequence),"static"))
    all_spcds.sort()
    start = int(all_spcds[0].split("_")[0])
    end = int(all_spcds[-1].split("_")[-1].split(".")[0])
    print(f'Start at {start}, end at {end}')

    for i in trange(start, end, frames):
        pcd = recoverLabels(root_dir, sequence, output_dir, i, i+frames, verbose=True, downSampleEvery=2)

    # TODO: Download ply files
