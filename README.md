
# BEV-Segmentation

## WIP

## Setup

### Clone Repository
```
git clone https://github.com/GreatGameDota/bev-segmentation.git
cd bev-segmentation
```  

### Install dependencies
```
pip install -r requirements.txt
git clone https://github.com/GreatGameDota/recoverKITTI360label
```  

## Download Data
#### Make sure to be in the `bev-segmentation` directory
### Download [KITTI-360 Dataset](http://www.cvlibs.net/datasets/kitti-360/)
```
sh data/download-KITTI-360/download_all.sh 2013_05_28_drive_0000_sync
```  

## Usage

### Accumulate Labeled Point Cloud Data

<img src="https://github.com/GreatGameDota/bev-segmentation/blob/main/.github/accum.png">

```
python accumulateLabeledPCD.py --sequence 2013_05_28_drive_0000_sync --frames 10 --drive
```

`python accumulateLabeledPCD.py -h` for more info on the command

### Generate Bird's Eye View images and segmentation maps

<img src="https://github.com/GreatGameDota/bev-segmentation/blob/main/.github/bev.png">

```
python generateImages.py --sequence 2013_05_28_drive_0000_sync --scale 10 --images --drive --data KITTI-360.yaml
```

`python generateImages.py -h` for more info on the command