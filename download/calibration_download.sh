# Calibration
root_dir=KITTI-360
data_3d_dir=.
zip_file=calibration.zip

cd $root_dir

curl https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/384509ed5413ccc81328cf8c55cc6af078b8c444/${zip_file} -o ${zip_file}
unzip -d ${data_3d_dir} ${zip_file}
rm ${zip_file}