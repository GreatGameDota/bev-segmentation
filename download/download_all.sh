# Download all data of given sequence

sh download/data_3d_raw_download.sh $1
sh download/data_2d_download.sh $1

sh download/data_3d_semantics_download.sh
sh download/data_3d_bboxes_download.sh
sh download/calibration_download.sh
sh download/data_poses_download.sh
