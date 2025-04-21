import deeplabcut
import os
import pandas as pd
import fnmatch
from typing import Tuple
import shutil
from pathlib import Path
import csv

class DeeplabcutProcessing:
    def __init__(
        self,
        config_path: str,
        root_dir: str,
        model_name: str,
        output_dir: str = None,
        single_file: bool = False,
        vis_root_dir: str = None,
        visualization: bool = False

        
    ):
        """
        config_path: path to the config files of the project
        root_dir: path to the videos. if single_file is true, point to .mp4 file
        model_name: name of the model used (this name will be used in output files)
        output_dir: path where the results should be stored. defaults to root_dir
        single_file: True when a single .mp4 needs anallysis. defaults to False.
        vis_root_dir: path where the visualizations should be stored. defaults to the output_dir
        visualization: whether visualization is required. defaults to False.
        """
        self.config_path = config_path
        self.root_dir = root_dir
        self.model_name = model_name
        self.output_dir = output_dir or root_dir
        self.single_file = single_file
        self.vis_root_dir = vis_root_dir or output_dir
        self.visualization = visualization

        # making directory for the raw DLC files
        self.dlc_path = os.path.join(self.output_dir, 'DLC_raw_files')
        os.makedirs(self.dlc_path, exist_ok=True)

        # checking whether input/output folders exist
        if not self.single_file:
            if not os.path.isdir(self.root_dir):
                raise FileNotFoundError(f"Root directory not found: {self.root_dir}")
        elif not self.root_dir.lower().endswith(('.mp4')):
           raise FileNotFoundError(f"'single_folder' is True, but 'root_dir' is not a single .mp4 file: {self.root_dir}")
               
        if not os.path.isdir(self.output_dir):
           raise FileNotFoundError(f"Output directory not found: {self.output_dir}")
        if not os.path.isdir(self.vis_root_dir):
           raise FileNotFoundError(f"Visualisation directory not found: {self.vis_root_dir}")
          
    def run(self):
        if self.single_file == True:
            video_list = [os.path.basename(self.root_dir)] # root dir points at single file
        else:
            video_list = sorted([
               f for f in os.listdir(self.root_dir)
               if f.lower().endswith(('.mp4'))
            ])

        for video_name in video_list:
            video_path = os.path.join(self.root_dir, video_name)

            deeplabcut.analyze_videos(self.config_path, video_path, destfolder= self.dlc_path, auto_track=False)
            deeplabcut.convert_detections2tracklets(self.config_path, video_path, destfolder=self.dlc_path)
            try:
                deeplabcut.stitch_tracklets(self.config_path, video_path, destfolder=self.dlc_path, videotype='mp4', save_as_csv = True)
            except:
                print(f"Skipping {video_name}: empty tracklets.")

            self._save_csv(video_name)

            if self.visualization:
                deeplabcut.create_video_with_all_detections(self.config_path, video_path, destfolder=self.dlc_path)
                video_name = video_name.replace('.mp4','')
                vis_filename = f'{video_name}_{self.model_name}_visualization.mp4'
                vis_path = os.path.join(self.vis_root_dir, vis_filename)
                
                dlc_path = Path(self.dlc_path) # converting to a Path object to be able to use glob
                for file in dlc_path.glob("*.mp4"):
                    shutil.move(str(file), vis_path)

       
    def _save_csv(self, video_name: str):
        video_name = video_name.replace('.mp4','')
        for file in os.listdir(self.dlc_path):
            if file.endswith(".csv") and fnmatch.fnmatch(file, f'{video_name}*'):
                full_path = os.path.join(self.dlc_path, file)
                df, output_filename = self._format_dlc_dataframe(full_path, video_name)
                output_path = os.path.join(self.output_dir, output_filename)
                # write headers
                header_0 = [self.model_name]
                header_1 = [video_name]

                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header_0)
                    writer.writerow(header_1)

                    # write headers from csv
                    for level in range(1, df.columns.nlevels):  # skip level 0 (scorer)
                        writer.writerow([col[level] if level < len(col) else '' for col in df.columns])
                    
                    # Write data rows
                    for row in df.itertuples(index=False, name=None):
                        writer.writerow(row)



    def _format_dlc_dataframe(self, file_path: str, video_name: str) -> Tuple[pd.DataFrame, str]:
        df = pd.read_csv(file_path, header=[0, 1, 2, 3])

        # Replace 'lower_neck' with 'neck' in the column MultiIndex
        df.columns = pd.MultiIndex.from_tuples([
            tuple('neck' if level == 'lower_neck' else level for level in col)
            for col in df.columns
        ])    

        # Format first column (frame numbers) as zero-padded .jpg
        df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: f"{int(float(x)):06d}.jpg" if pd.notnull(x) else x)

        output_filename = f"{video_name}_{self.model_name}_results.csv"
        return df, output_filename
