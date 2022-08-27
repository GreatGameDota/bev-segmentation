# Download all data of given sequence

root_dir=data/KITTI-360
mkdir -p $root_dir

sh data/download-KITTI-360/data_3d_raw_download.sh ${1:-"2013_05_28_drive_0002_sync"}
sh data/download-KITTI-360/data_2d_download.sh ${1:-"2013_05_28_drive_0002_sync"}

sh data/download-KITTI-360/data_3d_semantics_download.sh
sh data/download-KITTI-360/data_3d_bboxes_download.sh
sh data/download-KITTI-360/calibration_download.sh
sh data/download-KITTI-360/data_poses_download.sh
