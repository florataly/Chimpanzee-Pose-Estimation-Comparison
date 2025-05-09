# Multi-Model Pose Estimation & Evaluation Toolkit (ASAB 2025 Spring)

## Installation

Please note that NVIDIA GPUs are needed for the replication of this project.

First clone this project:
```
git clone https://github.com/florataly/Chimpanzee-Pose-Estimation-Comparison
cd Chimpanzee-Pose-Estimation-Comparison
```

To compare DeppLabCut and MMPose estimations, we need separate environments for each. We recommend using Anaconda virtual environments, as MMPose runs more reliably here.

### [MMPose v0.26](https://github.com/open-mmlab/mmpose/tree/v0.26.0) Environment

```
cd OpenApePose

conda create -n MMPose026 python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate MMPose026

pip install openmim==0.3.3
mim install mmcv-full==1.6.0

git clone --branch v0.26.0 https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -e .

cd ..

# the old version of mmdet could be installed using 'mim', but it can crash out easily, so I'm installing from source
git clone --branch v2.25.0 https://github.com/open-mmlab/mmdetection.git 
cd mmdetection
pip install -v -e .

pip install tomli==2.0.1
pip install platformdirs==3.5.1
pip install tqdm

cd ..
```

Run DEMO to confirm successful installation, the demo images should appear in mmpose/vis_results folder. If you run into trouble installing MMPose, please follow branch v0.26.0 of OpenPose for guidance.
```
conda activate MMPose026
cd OpenApePose/mmpose
python demo/top_down_img_demo.py configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth --img-root tests/data/coco/ --json-file tests/data/coco/test_coco.json --out-img-root vis_results
cd ../..
```

Follow the installation guides of [OpenApePose](https://github.com/desai-nisarg/OpenApePose) to be able to use their model. This requires the download of their model weights and codes and placement of these into the correct folders within the mmpose folder we created.

### DeepLabCut Environment
Follow the installation guides of [DeepLabCut](https://deeplabcut.github.io/DeepLabCut/docs/installation.html).

For DeepWild, please follow the google drive link shared on [DeepWild](https://github.com/Wild-Minds/DeepWild)'s repository. Then place the downloaded file to the DLC folder.

If the DLC environment will be used for evaluation as well, the following packages are further required:
```
pip install pandas numpy scipy statsmodels matplotlib seaborn
```

## Folder structure
The demo scripts expect the following folder structure, so please create the missing folders and add the required data, if you wish to run the demo scripts:
```
Chimpanzee-Pose-Estimation-Comparison
|-- data
    │-- ground truth
    |-- TEST
├-- DLC
    |--DeepWild1.1
├-- OpenApePose
    |-- mmpose
        |--checkpoints
        |-- ...
├-- predictions
├-- evaluation
    |-- results
```

## Data
In Data/TEST the scripts expect to find .mp4 videos or folders with the same name as the .mp4 video files, including the frames of the video. \\
In the Ground Truth folder we expect to see the manually labelled .csv files of each video in DLC style. Please find tools in the /tool folder to convert json files to this format.

The csv structure should be the follownig (find `data/Ground Truth/demo.csv` file for reference):
```
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
```

 ## Pose estimations

### OpenApePose
 First we run OpenApePose on folders containing frames of a video. Example usage in the terminal:
 (please beware, this won't run unless all installations have been successful)
```
cd OpenApePose
conda activate MMPose026
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
Personally, I suggest using Jupyter notebook. Please find detailed explanation of each function in the OpenApePose/MMpose_procressing.ipynb file.

### DeepWild
Second, we process the .mp4 videos with DeepWild. Example usage in the terminal:
(please beware, this won't run unless all installations have been successful)
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
The evaluation scripts can be run in the DLC environment, if the additional packages have been installed as described above. The resulting files should include an individual .csv file for each video separately, a summary csv when save_summary is True, regression_results.txt if environmental factors have been taken into consideration, and images of the distributions of PCK, MPJPE and matched frames for each model. Please find a demo of these files in the 'demo results' folder.
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

