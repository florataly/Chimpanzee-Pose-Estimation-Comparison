import os
import csv
import cv2
from collections import defaultdict
from mmcv import Config
from mmpose.apis import (
    init_pose_model,
    inference_top_down_pose_model,
    vis_pose_result,
    process_mmdet_results
)
from mmdet.apis import init_detector, inference_detector
from pathlib import Path
from tqdm import tqdm
from torchvision.ops import nms
import torch



class MMPose026Processing:
    def __init__(
        self,
        pose_config_path: str,
        root_dir: str,
        model: str,
        output_dir: str = None,
        vis_root_dir: str = None,
        visualization: bool = False,
        single_folder: bool = False,
        process_every_n: int = None,
        pose_checkpoint_path: str = './mmpose/checkpoints/hrnet_w48_oap_256x192_full.pth',
        det_config_path: str = './mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
            # checkpoint file from https://github.com/open-mmlab/mmdetection/blob/v2.26.0/demo/inference_demo.ipynb
        det_checkpoint_path: str = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    ):
        """
        pose_config_path: path to the config files
        pose_checkpoint_path: path to the checkpoint files
        root_dir: path to the folder with images
        model: name of the model. this name will be used when saving the results into CSV files.
        output_dir: path where the results will be stored. defaults to root_dir.
        visualization: whether visualization is required. defaults to false.
        vis_root_dir: path where the visualizations will be saved. defaults to the output_dir
        single_folder: whether the frames are in a single folder or a set of folders (i.e one folder per video). defaults to false.
        process_every_n: when set to integer 'n', the model will only process every n-th image (sorted by name).
        det_config_path: path to MMdet config files. defaults to the demo files provided by MMpose
        det_checkpoint_path: path to MMdet checkpoint files. defaults to the demo files provided by MMpose
        """
        self.pose_config_path = pose_config_path
        self.pose_checkpoint_path = pose_checkpoint_path
        self.root_dir = root_dir
        self.model = model
        self.output_dir = output_dir or root_dir
        self.vis_root_dir = vis_root_dir or output_dir
        self.visualization = visualization
        self.det_config_path = det_config_path
        self.det_checkpoint_path = det_checkpoint_path
        self.single_folder = single_folder
        self.process_every_n = process_every_n


        # checking whether input/output folders exist
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")
        if not os.path.isdir(self.output_dir):
            raise FileNotFoundError(f"Output directory not found: {self.output_dir}")
        if not os.path.isdir(self.vis_root_dir):
            raise FileNotFoundError(f"Visualisation directory not found: {self.vis_root_dir}")

        #init for apis (from MMpose apis library)
        self.pose_model = init_pose_model(self.pose_config_path, self.pose_checkpoint_path)
        self.det_model = init_detector(self.det_config_path, self.det_checkpoint_path)

        # getting dataset config path from the provided config file
        dataset_cfg = Config.fromfile(self._extract_dataset_config_path())

        # getting keypoint info from the dataset config file
        self.bodyparts, self.num_keypoints = self._extract_keypoint_info(dataset_cfg)

    def _extract_dataset_config_path(self):
        with open(self.pose_config_path, 'r') as f:
            base_lines = []
            inside_base = False
            for line in f:
                line = line.strip()
                # file location should be in the __base__ section
                if line.startswith('_base_'):
                    inside_base = True
                    continue
                if inside_base:
                    if ']' in line:
                        inside_base = False
                    else:
                        base_lines.append(line.strip(',').strip("'").strip('"').strip("]"))
        # in MMpose 0.26 there is always a default_runtime.py here next to the dataset config file
        for path in base_lines:
            if 'default_runtime.py' not in path and path.endswith('.py'):
                return os.path.normpath(os.path.join(os.path.dirname(self.pose_config_path), path))

        raise FileNotFoundError("Could not find dataset config path in _base_ block.")


    def _extract_keypoint_info(self, cfg):
        # the keypoints a model uses are listed in the dataset config file
        keypoint_info = cfg.dataset_info['keypoint_info']
        bodyparts = [v['name'] for _, v in sorted(keypoint_info.items())]
        return bodyparts, len(bodyparts)

    def run(self):
        # when all pictures are stored in a single folder
        if self.single_folder:
            subdirs = [('', self.root_dir)]
        
        # if there are several folders with frames, we loop through each folder 
        # and store the results in individual folders as well
        else:
            subdirs = [
                (name, os.path.join(self.root_dir, name))
                for name in sorted(os.listdir(self.root_dir))
                if os.path.isdir(os.path.join(self.root_dir, name))
            ]

        for folder, folder_path in subdirs:
            self._process_folder(folder, folder_path)

    def _process_folder(self, folder: str, folder_path: str):
        image_animal_data = defaultdict(dict)

        # creating visualization output folders (vis_dir)
        if self.visualization:
            os.makedirs(os.path.join(self.vis_root_dir, folder), exist_ok=True)
            vis_dir = os.path.join(self.vis_root_dir, folder)

        # only working on image files
        image_list = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        # processing only every 'n'th image (if process_every_n is not none)
        if self.process_every_n is not None:
            image_list = image_list[::self.process_every_n]

        # looping through images in a single folder
        for i, img_name in enumerate(tqdm(image_list, desc=f"Processing {folder}", unit="frame")): #progress bar
            img_path = os.path.join(folder_path, img_name)
            
            # MMdet inference
            mmdet_results = inference_detector(self.det_model, img_path)
            if not mmdet_results:
                continue

            # only keeping the animal (including human) detections. 
            # BEWARE process_mmdet_results deducts one from the IDs
            # 0: person; 14:bird; 15:cat; 16:dog; 17:horse; 18:sheep; 19:cow; 20:elephant; 21:bear; 22:zebra; 23:giraffe
            animal_results = []
            for cat_id in [1, 17, 18, 19, 20, 21, 22, 23, 24]:
                animal_results.extend(process_mmdet_results(mmdet_results, cat_id=cat_id))

            pose_results, _ = inference_top_down_pose_model(
                self.pose_model, img_path, animal_results, bbox_thr=0.3, format='xyxy',
                dataset=self.pose_model.cfg.data.test.type)
            
            # non-maximum suppression (remove high Itersection over Union boxes)
            boxes = []
            scores = []
            for i in range(len(pose_results)):
                boxes.append(pose_results[i]['bbox'][:4].tolist())
                scores.append(pose_results[i]['bbox'][4])

            boxes = torch.tensor(boxes)
            scores = torch.tensor(scores)

            keep = nms(boxes, scores, iou_threshold = 0.85) # IoU threshold can be changed
            pose_results = [pose_results[i] for i in keep.tolist()]

            # if there are no results, we create empty results for one animal
            if not pose_results:
                image_animal_data[img_name][f'animal_0'] = [['', '', '']] * self.num_keypoints

            # saving x, y, score along with how many animals we see in a single image
            for i, result in enumerate(pose_results):
                keypoints = [[x, y, score] for x, y, score in result['keypoints']]
                image_animal_data[img_name][f'animal_{i}'] = keypoints

            # visualizing results into visualization output folder (vis_dir)
            if self.visualization:
                vis_result = vis_pose_result(
                    self.pose_model, img_path, pose_results,
                    dataset=self.pose_model.cfg.data.test.type, show=False)
                cv2.imwrite(os.path.join(vis_dir, f'{os.path.splitext(img_name)[0]}_pose.jpg'), vis_result)
                
        # creating video if we didn't skip over any frames
        if self.visualization and self.process_every_n == None:
            self.images_to_videos(input_dir=vis_dir, single_folder = True)

        self._save_csv(folder, image_animal_data)

    def _save_csv(self, folder: str, image_animal_data: dict):
        # checking what's the max number of individuals on a single image for CSV formating
        max_animals = max((len(animals) for animals in image_animal_data.values()), default=0)

        # first row has the name of the video
        header_0 = [self.model]
        header_1 = [folder]
        header_2, header_3, header_4 = ['individuals'], ['bodyparts'], ['coords']
        
        # each individual has x, y, and likelihood for each bodypart. 
        # this is repeated as many times as the number of max individuals on a single picture
        for a_id in range(max_animals):
            for bp in self.bodyparts:
                header_2 += [f'individual{a_id+1}'] * 3
                header_3 += [bp] * 3
                header_4 += ['x', 'y', 'likelihood']

        # putting all data from a single image into a single line, sorted by the animal ID
        data_rows = []
        for img in sorted(image_animal_data.keys()):
            row = [img]
            for a_id in range(max_animals):
                aid_key = f'animal_{a_id}'
                if aid_key in image_animal_data[img]:
                    for kp in image_animal_data[img][aid_key]:
                        row.extend(kp)
                else:
                    row.extend([''] * self.num_keypoints * 3)
            data_rows.append(row)

        # writing csv
        csv_path = os.path.join(self.output_dir, f'{folder}_{self.model}_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header_0)
            writer.writerow(header_1)
            writer.writerow(header_2)
            writer.writerow(header_3)
            writer.writerow(header_4)
            writer.writerows(data_rows)

        print(f"CSV saved: {csv_path}")


    def videos_to_images(self, input_folder = None, output_dir = None):
        """
        input_folder: where the video(s) are. defaults to root_dir
        output_dir: where the frames should be saved to. defaults to the input_folder.
        """
        if not input_folder:
            input_folder = Path(self.root_dir)
        else:
            input_folder = Path(input_folder)
        
        if not output_dir:
            output_dir = Path(input_folder)
        else:
            output_dir = Path(output_dir)

        # cv2's supported video file extensions
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}

        #looping through all video files in folder with correct suffix
        for video_file in input_folder.iterdir():
            if video_file.suffix.lower() in video_extensions:
                video_name = video_file.stem
                
                # creating a folder for each video
                output_folder = output_dir / video_name
                output_folder.mkdir(exist_ok=True)

                cap = cv2.VideoCapture(str(video_file))
                frame_num = 0 #numbering the frames

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_filename = output_folder / f"{frame_num:06d}.jpg" #matches ChimpACT's frame names
                    cv2.imwrite(str(frame_filename), frame)
                    frame_num += 1
                
                cap.release()
                print(f"extracted {frame_num} frames from {video_file.name} to {output_folder}")
    
    def images_to_videos(self, input_dir = None, output_folder = None, single_folder = False, fps = 30):
 
        """
        input_dir: where the frames or folders of frames are stored. defaults to root_dir
        output_folder: where the video(s) will be saved. defaults to input_dir
        single_folder: True when input_dir is a single folder with image files within. False when inout_dir points at a folder with further folders - each folder will turn into an individual video.
        fps: frames per second. defaults to 30
        """
        if not input_dir:
            input_dir = Path(self.root_dir)
        else:
            input_dir = Path(input_dir)
        
        if not output_folder:
            output_folder = Path(input_dir)
        else:
            output_folder = Path(output_folder) 

        # if there is only a single folder
        if single_folder:
            folders = [input_dir]
        else:
            folders = sorted([f for f in input_dir.iterdir() if f.is_dir()])

        for folder_path in folders:
            folder_name = folder_path.name
            output_path = output_folder / f"{folder_name}.mp4"

            # skip video if it already exists
            if output_path.is_file():
                print(f"Skipping '{folder_name}' - video already exists.")
                continue

            # sorting image files into list
            images = sorted([img for img in folder_path.iterdir() if img.suffix.lower() in {'.png', '.jpg', '.jpeg'}])

            #skipping folder if there are no images there
            if not images:
                print(f"Skipping '{folder_name}' - no image files found")
                continue

            first_frame = cv2.imread(str(images[0]))
            height, width, _ = first_frame.shape

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            for img_path in images:
                frame = cv2.imread(str(img_path))
                out.write(frame)

            out.release()
            print(f"Saved video to: {output_path}")
        
        print("All videos processed.")





