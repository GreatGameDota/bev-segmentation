
train_list="$1"

root_dir=data/KITTI-360
data_3d_dir=data_3d_raw

mkdir -p $root_dir/$data_3d_dir

cd $root_dir 

# 3d scans
for sequence in ${train_list}; do
    zip_file=${sequence}_velodyne.zip
    curl https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_3d_raw/${zip_file} -o ${zip_file}
    unzip -d ${data_3d_dir} ${zip_file} 
    rm ${zip_file}
done

# timestamps
zip_file=data_timestamps_velodyne.zip
curl https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_3d_raw/${zip_file} -o ${zip_file}
unzip -d ${data_3d_dir} ${zip_file}
rm $zip_file