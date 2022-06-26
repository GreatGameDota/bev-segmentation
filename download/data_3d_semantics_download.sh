# Labeled Point Clouds
root_dir=KITTI-360
data_3d_dir=.
zip_file=data_3d_semantics.zip

cd $root_dir

curl https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/6489aabd632d115c4280b978b2dcf72cb0142ad9/${zip_file} -o ${zip_file}
unzip -d ${data_3d_dir} ${zip_file}
rm ${zip_file}