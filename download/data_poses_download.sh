# Vehicle Pose
root_dir=KITTI-360
data_3d_dir=data_poses
zip_file=data_poses.zip

cd $root_dir

curl https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/${zip_file} -o ${zip_file}
unzip -d ${data_3d_dir} ${zip_file}
rm ${zip_file}