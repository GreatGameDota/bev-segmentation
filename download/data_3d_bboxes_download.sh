%%bash
# 3d Bboxes
root_dir=KITTI-360
data_3d_dir=.
zip_file=data_3d_bboxes.zip

cd $root_dir

curl https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/ffa164387078f48a20f0188aa31b0384bb19ce60/${zip_file} -o ${zip_file}
unzip -d ${data_3d_dir} ${zip_file}
rm ${zip_file}