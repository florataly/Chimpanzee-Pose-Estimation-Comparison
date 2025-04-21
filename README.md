# Multi-Model Pose Estimation & Evaluation Toolkit (ASAB 2025 Spring)

## Installation

To compare DeppLabCut and MMPose estimations, we need separate environments for each. We recommend using Anaconda virtual environments, as MMPose runs more reliably here.

### MMPose_0.26 Environment

```
cd OpenApePose
conda create -n MMPose026 python=3.8 pytorch=1.10 torchvision -c pytorch –y
conda activate MMPose026

pip install openmim==0.3.3
pip install numpy==1.24
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
mim install mmdet== 2.25.0

pip install matplotlib==3.7
pip install pillow==9.3.0
pip install xtcocotools==1.13
pip install tomli==2.0.1
pip install platformdirs==3.5.1

git clone --branch v0.26.0 https://github.com/open-mmlab/mmpose.git 
cd mmpose
pip install -e .
cd ..
```

Run DEMO to confirm successful installation. If you run into trouble installing MMPose, please follow branch v0.26.0 of OpenPose for guidance.
```
conda activate OpenApePose
cd OpenApePose/mmpose
python demo/top_down_img_demo.py \ 
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \ 
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \ 
    --img-root tests/data/coco/ --json-file tests/data/coco/test_coco.json \ 
    --out-img-root vis_results
cd ../..
```

Follow the installation guides of OpenApePose to be able to use their model. This requires the download of their model weights and codes and placement of these into the correct folders within MMPose.

### DeepLabCut Environment
Follow the installation guides of (DeepLabCut)[https://deeplabcut.github.io/DeepLabCut/docs/installation.html].
For DeepWild, please follow the google drive link shared on (DeepWild)[https://github.com/Wild-Minds/DeepWild]'s repository. Then place the downloaded file to the DLC folder.

## Data
In Data/TEST the scripts expect to find .mp4 videos or folders with the same name as the .mp4 video files, including the frames of the video. \\
In the Ground Truth folder we expect to see the manually labelled .csv files of each video in DLC style. Please find tools in the /tool folder to convert MMPose json to this format.

The file structure should be the follownig (find `data/Ground Truth/demo.csv` file for reference):

| video name                      |             |
| individuals                     | individual0 | individual0 | individual0 | individual0 | individual0 | . . . | individual1 | . . .
| bodyparts                       | hip         | hip         | hip         | right_knee  | right_knee  |       | hip         |
| coords                          | x           | y           | visibility  | x           | y           |       | x           | 
|---------------------------------|-------------|-------------|-------------|-------------|-------------|       |-------------| 
| 000000.jpg                      | 359.912045  | 444.159332  | 2           | 393.514307  | 416.417065  |       | 301.560284  |
| 000001.jpg                      | 361.742193  | 445.460088  | 2           | 394.369693  | 417.272451  |       | 300.732491  |
| 000002.jpg                      | 364.044463  | 444.502356  | 2           | 393.183189  | 413.119689  |       | 303.69875   |
| 000003.jpg                      | 361.878773  | 448.586566  | 2           | 393.183189  | 413.119689  |       | 303.105498  |
 . . . 

 ## Pose estimations

 First we run OpenApePose. Example in the terminal:
 (please beware, this won't run unless all installations have been successful before)
```
cd OpenApePose
conda activate OpenApePose
python3
from mm_pose_processing import MMPose026Processing

inferencer = MMPose026Processing(
    pose_config_path='./mmpose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/macaque/hrnet_w48_oap_256x192_full.py',
    pose_checkpoint_path='./OpenApePose/mmpose/checkpoints/oap/hrnet_w48_oap_256x192_full.pth',
    root_dir='../data/',
    model='oap',
    output_dir='../predictions/',
    visualization = True
)
inferencer.run()
exit()
cd ..
```
Personally, I suggest using Jupyter notebook for running the inferencers rather than in the terminal. Please find detailed explanation of each function in the MMpose_procressing.ipynb file.


Then we process DLC. Example in the terminal:
(please beware, this won't run unless all installations have been successful before)
```
cd DLC
conda activate DEEPLABCUT
python3
from from dlc_processing import DeeplabcutProcessing

inferencer = DeeplabcutProcessing(
    config_path='./DeepWild1.1/deepwild2-Charlotte-2023-05-24/config.yaml',
    root_dir='../data/',
    output_dir='../predictions/',
    model_name='deepwild'
)

inferencer.run()
exit()
cd ..
```

## Evaluation
The evaluation should be able to run in the DLC environment. The resulting files should include an individual .csv file for each video separately, a summary csv when save_summary is True, regression_results.txt if environmental factors have been taken into consideration, and images of the distributions of PCK, MPJPE and matched frames for each model. 
Example usage in the terminal:
```
cd evluation
conda activate DEEPLABCUT
python3
from evaluation import ComparePoseModels

inferencer = ComparePoseModels(
    predictions_dir='./predictions/',
    gt_dir='./data/ground truth',
    output_dir='./evaluation/results/',
    scale=True,
    pck_treshold=0.25
)

inferencer.evaluate_videos()
inferencer.compare_models(save_summary=True)
inferencer.visualisation()
exit()
cd ..
```

