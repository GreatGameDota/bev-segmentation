
train_list="$1"
cam_list="00"

root_dir=KITTI-360
data_2d_dir=data_2d_raw

mkdir -p $root_dir
mkdir -p $root_dir/$data_2d_dir

cd $root_dir 

# perspective images
for sequence in ${train_list}; do
    for camera in ${cam_list}; do 
	zip_file=${sequence}_image_${camera}.zip
        curl https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_2d_raw/${zip_file} -o ${zip_file}
	unzip -d ${data_2d_dir} ${zip_file} 
	rm ${zip_file}
    done
done

# timestamps
zip_file=data_timestamps_perspective.zip
curl https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_2d_raw/${zip_file} -o ${zip_file}
unzip -d ${data_2d_dir} ${zip_file}
rm $zip_file