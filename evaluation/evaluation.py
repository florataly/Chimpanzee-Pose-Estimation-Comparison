import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import os
import csv
from io import StringIO
from itertools import combinations
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches


class ComparePoseModels:
    def __init__(
        self,
        predictions_dir: str,
        gt_dir: str,
        output_dir: str = None,
        single_file: bool = False,
        pck_treshold: int = None,
        scale: bool = True

    ):
        """
        predictions_dir: folder where prediction results are. if single_file is true, then path to specific CSV
        gt_dir: folder where ground truth CSVs are. if single_file is true, then path to specific CSV
        output_dir: folder where evaluation results will be stored. defaults to prediction_dir
        single_file: True when there is a single file that needs evaluating. if True, prediction and gt directories need to point to the specific file. defaults to False
        pck_treshold: treshold of how close the prediction needs to be to the ground truth to count as correct prediction. when scale = False defaults to 10 pixels, when scale = True defaults to 0.5 (= 50%) of head bone link
        scale: True when evaluation measures with respect to the scale of the subject. False when evaluation measures with predefined pixel distance. defaults to True
        """
        self.predictions_dir = predictions_dir
        self.gt_dir = gt_dir
        self.output_dir = output_dir or predictions_dir
        self.single_file = single_file
        self.pck_treshold = pck_treshold
        self.scale = scale

        # checking whether input/output folders exist
        if not os.path.isdir(self.predictions_dir):
            raise FileNotFoundError(f"Predictions path not found: {self.predictions_dir}")
        if not os.path.isdir(self.output_dir):
            raise FileNotFoundError(f"Output directory not found: {self.output_dir}")
        if not os.path.isdir(self.gt_dir):
            raise FileNotFoundError(f"Ground truth path not found: {self.gt_dir}")

    def _load_csv(self, csv_path: str, is_gt: bool):
        with open(csv_path, 'r') as f:
            lines = f.readlines()

        # model name from [0]
        if not is_gt:
            model = lines[0].strip().split(',')[0]
            lines = lines[1:]

        
        # videoname from [1]
        video_name = lines[0].strip().split(',')[0]

        # headers
        row1 = lines[1].strip().split(',')
        row2 = lines[2].strip().split(',')
        row3 = lines[3].strip().split(',')

        col_labels = []
        for i in range (1, len(row1)):
            col_labels.append((row1[i], row2[i], row3[i]))

        # data starts at lines[4]
        data_lines = lines[4:]
        index=[] #fame name
        values=[]  #data

        for line in data_lines:
            parts = line.strip().split(',')
            if len(parts) < 2: #if there is no data after an index we skip line
                continue
            index.append(parts[0])
            values.append(parts[1:])
        
        df = pd.DataFrame(values, index=index, columns=pd.MultiIndex.from_tuples(col_labels))
        df = df.apply(pd.to_numeric, errors='coerce')
        
        if is_gt:
            return df, video_name
        else:
            return df, model, video_name
    
    def _extract_individuals(self, frame_data, keypoints, is_gt: bool):
        individuals = []
        individuals_in_frame = frame_data.columns.get_level_values(0).unique()
        field_name = 'likelihood' if not is_gt else 'visibility'

        for ind in individuals_in_frame:
            kp_coords = {}
            for kp in keypoints:
                try:
                    x = frame_data[(ind, kp, 'x')].values[0]
                    y = frame_data[(ind, kp, 'y')].values[0]
                    score_or_vis = frame_data[(ind, kp, field_name)].values[0]
                    valid = score_or_vis > 0.1 # confidence must be over 0.1
                    kp_coords[kp] = (x, y) if valid else (np.nan, np.nan)
                except KeyError:
                    continue # no valid keypoint present
            individuals.append(kp_coords)
        return individuals
    
    # === Mean Per Joint Position Error (MPJPE): Average distance error from ground truth ===
    def _mpjpe(self, pred_dict, gt_dict):
        distances = []
        shared_kps = set(pred_dict.keys()) & set(gt_dict.keys())

        for kp in shared_kps:
            px, py = pred_dict[kp]
            gx, gy = gt_dict[kp]
            if not np.isnan(px) and not np.isnan(gx):
                d = np.linalg.norm([px - gx, py - gy]) # pixel distance
                distances.append(d)

        if len(distances) == 0: # when there were no valid predictios, returning 1000000
            return 1e6
        return np.mean(distances) 
    
    def _mpjpe_scaled(self, pred_dict, gt_dict, neck_hip_dict):
        # measuring the head bone link in ground truth
        neck_x, neck_y = neck_hip_dict['neck']
        hip_x, hip_y = neck_hip_dict['hip']

        if not np.isnan(hip_x) and not np.isnan(neck_x):
            neck_hip_d = np.linalg.norm([neck_x-hip_x, neck_y-hip_y])
        else: # no neck-hip distance in ground truth
            return 1e6

        distances = []
        shared_kps = set(pred_dict.keys()) & set(gt_dict.keys())

        for kp in shared_kps:
            px, py = pred_dict[kp]
            gx, gy = gt_dict[kp]
            if not np.isnan(px) and not np.isnan(gx):
                d = np.linalg.norm([px - gx, py - gy]) # pixel distance
                distances.append(d/neck_hip_d) # pixel distance compared to the head bone link

        if len(distances) == 0: # when there were no valid predictios, returning 1000000
            return 1e6
        return np.mean(distances) 
    
    # === Percentage of Correct Keypoints (PCK): % of shared keypoints within 10px of GT ===
    def _pck(self, pred_dict, gt_dict, threshold=10.0): # default treshold is 10 pixels
        correct = 0
        total = 0
        shared_kps = set(pred_dict.keys()) & set(gt_dict.keys())

        for kp in shared_kps:
            px, py = pred_dict[kp]
            gx, gy = gt_dict[kp]
            if not np.isnan(px) and not np.isnan(gx):
                d = np.linalg.norm([px - gx, py - gy])
                total += 1
                if d <= threshold:
                    correct += 1

        if total == 0:
            return None
        return correct / total
    
    def _pck_scaled(self, pred_dict, gt_dict, neck_hip_dict, threshold=0.5): # default treshold is 50% of head bone link
        # measuring the head bone link in ground truth
        neck_x, neck_y = neck_hip_dict['neck']
        hip_x, hip_y = neck_hip_dict['hip']
        if not np.isnan(hip_x) and not np.isnan(neck_x):
            neck_hip_d = np.linalg.norm([neck_x-hip_x, neck_y-hip_y])
        else: # no neck-hip distance
            return None
        
        correct = 0
        total = 0
        shared_kps = set(pred_dict.keys()) & set(gt_dict.keys())

        for kp in shared_kps:
            px, py = pred_dict[kp]
            gx, gy = gt_dict[kp]
            if not np.isnan(px) and not np.isnan(gx):
                d = np.linalg.norm([px - gx, py - gy])
                total += 1
                if d <= (threshold*neck_hip_d): # is the distance smaller than 'treshold'% of head bone link?
                    correct += 1

        if total == 0:
            return None
        return correct / total
    

    def _evaluate(self, pred_csv, gt_csv):
        pred_df, model, video_pred = self._load_csv(pred_csv, is_gt=False)
        gt_df, video_gt = self._load_csv(gt_csv, is_gt=True)
        assert video_pred == video_gt, f"Mismatch: predicted video = {video_pred}, gt video ={video_gt}"
        video_name = video_pred

        keypoints = list(set(pred_df.columns.get_level_values(1)))
        results = []
        valid_frames = 0

        # getting frames with both predictions and ground truth on them
        frames_to_evaluate = pred_df.index.intersection(gt_df.index)

        for frame in frames_to_evaluate:
            pred_frame = pred_df.loc[[frame]]
            gt_frame = gt_df.loc[[frame]]

            pred_inds = self._extract_individuals(pred_frame, keypoints, is_gt=False)
            gt_inds = self._extract_individuals(gt_frame, keypoints, is_gt=True)
            
            # getting indices for 'neck' and 'hip' from ground truth to calculate head bone link
            if self.scale:
                try:
                    neck_hip_inds = self._extract_individuals(gt_frame, ['neck', 'hip'], is_gt=True)
                except KeyError:
                    print(f"No 'neck' and 'hip' keypoints found in the ground truth. Please set 'scale' to False or rename ground truth!")
            
            # Skip if all predictions are nan
            if all(all(np.isnan(x) or np.isnan(y) for (x, y) in ind.values()) for ind in pred_inds):
                continue
            if len(pred_inds) == 0 or len(gt_inds) == 0:
                continue

            valid_frames += 1

            cost_matrix = np.zeros((len(pred_inds), len(gt_inds)))
            for i, pred in enumerate(pred_inds):
                for j, gt in enumerate(gt_inds):
                    if self.scale:
                        cost_matrix[i, j] = self._mpjpe_scaled(pred_dict=pred, gt_dict=gt, neck_hip_dict=neck_hip_inds[j])
                    else:
                        cost_matrix[i, j] = self._mpjpe(pred_dict=pred, gt_dict=gt)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            frame_result = {
                'frame': frame,
                'mpjpe_scores': [],
                'pck_scores': [],
                'matches': []
            }

            for r, c in zip(row_ind, col_ind):
                pred = pred_inds[r]
                gt = gt_inds[c]

                if self.scale: # scaling to head bone link
                    mpjpe_score = self._mpjpe_scaled(pred_dict=pred, gt_dict=gt, neck_hip_dict=neck_hip_inds[c])
                    if self.pck_treshold:
                        pck_score = self._pck_scaled(pred_dict=pred, gt_dict=gt, neck_hip_dict=neck_hip_inds[c], threshold=self.pck_treshold)
                    else:
                        pck_score = self._pck_scaled(pred_dict=pred, gt_dict=gt, neck_hip_dict=neck_hip_inds[c])
                else: # not scaling to head bone link
                    mpjpe_score = self._mpjpe(pred_dict=pred, gt_dict=gt)
                    if self.pck_treshold:
                        pck_score = self._pck(pred_dict=pred, gt_dict=gt, threshold=self.pck_treshold)
                    else:
                        pck_score = self._pck(pred_dict=pred, gt_dict=gt)

                frame_result['mpjpe_scores'].append(mpjpe_score)
                frame_result['pck_scores'].append(pck_score)
                frame_result['matches'].append((r, c))

            results.append(frame_result)
        
        print(f"Evaluated {valid_frames} frames with usable predictions.")
        return results, video_name
    
    def _export_matches_to_csv(self, results, model, videoname):
        output_dir = Path(self.output_dir)

        # evaluation file name
        if self.scale:
            output_file = output_dir / f"{videoname}_{model}_scaled_evaluation.csv"
        else:
            output_file = output_dir / f"{videoname}_{model}_evaluation.csv"

        rows = []
        for frame_result in results:
            frame = frame_result['frame']
            for (pred_idx, gt_idx), mpjpe_score, pck_score in zip(
                frame_result['matches'],
                frame_result['mpjpe_scores'],
                frame_result['pck_scores']
            ):
                rows.append({
                    'frame': frame,
                    'predicted_individual_index': pred_idx,
                    'ground_truth_individual_index': gt_idx,
                    'mpjpe': mpjpe_score,
                    'pck': pck_score
                })
        df_out = pd.DataFrame(rows)
        
        # Drop rows where "pck" is NaN or lower than 0.01
        df_out = df_out[
            (df_out["pck"].notna()) &
            (df_out["pck"] >= 0.01)
        ]

        # header
        header_0 = [model]
        header_1 = [videoname]

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header_0)
            writer.writerow(header_1)
            
            #columns
            writer.writerow(df_out.columns)

            for row in df_out.itertuples(index=False, name=None):
                writer.writerow(row)

        print(f"Exported matching results to: {output_file}")

    def evaluate_videos(self):
        pred_dir = Path(self.predictions_dir)
        gt_dir = Path(self.gt_dir)

        if self.single_file:
            pred_files = [os.path.basename(pred_dir)] # it's a file path not a folder
        else:
            pred_files = list(pred_dir.glob("*_results.csv"))

        for pred_file in pred_files:
            try:
                _, model, video_name = self._load_csv(pred_dir / pred_file.name, is_gt=False)
            except Exception as e:
                print(f"Could not extract model and video name from: {pred_file.name}, error: {e}")
                continue
            
            gt_file = next(gt_dir.glob(f"*{video_name}*.csv"), None)

            if gt_file is None or not gt_file.exists():
                print(f"Ground truth file not found for: {video_name}")
                continue

            print(f"Evaluating video: {video_name}, model: {model}")
            try:
                results, videoname = self._evaluate(pred_file, gt_file)
                self._export_matches_to_csv(results, model, videoname)
            except Exception as e:
                print(f"Error evaluating {video_name}: {e}")

    def _summary(self, output_dir, save_as_csv):
        summary_data = []
        individual_data = []
        
        eval_dir = Path(self.output_dir)
        eval_files = list(eval_dir.glob("*_evaluation.csv"))

        for file in eval_files:
            try:
                with open(file, 'r') as f:
                    lines = f.readlines()

                # model and video_name from first two lines
                model = lines[0].strip().split(',')[0]
                video_name = lines[1].strip().split(',')[0]

                csv_data = ''.join(lines[2:])  # Join remaining lines
                df = pd.read_csv(StringIO(csv_data))     

            except Exception as e:
                print(f"Error reading {file}: {e}")      

            for idx, row in df.iterrows(): 
                individual_data.append({
                    'model': model,
                    'video': video_name,
                    'frame_index': row['frame'],
                    'pck': row['pck'],
                    'mpjpe': row['mpjpe']
                })

            # Compute summary metrics

            pck_mean = df['pck'].mean()
            mpjpe_mean = df['mpjpe'].mean()
            matched_frames = len(df)

            summary_data.append({
                'model': model,
                'video': video_name,
                'pck_mean': pck_mean,
                'mpjpe_mean': mpjpe_mean,
                'matched_frames': matched_frames
            })

        df_summary = pd.DataFrame(summary_data).dropna()
        df_individual = pd.DataFrame(individual_data).dropna()

        if save_as_csv:
            output_dir = Path(output_dir)
            df_summary.to_csv(output_dir / "summary.csv", index=False)
            print(f"Summary and individual metric CSVs saved to {output_dir}")
        
        return(df_summary, df_individual)
    
    def compare_models(self, output_dir = None, save_summary = False):
        if not output_dir:
            output_dir = self.output_dir
        
        df, _ = self._summary(output_dir, save_as_csv=save_summary)

        # Mann-Whitney U tests between model summaries
        metrics = ['pck_mean', 'mpjpe_mean', 'matched_frames']
        results = []

        models = df['model'].unique()
        model_pairs = list(combinations(models, 2))

        for metric in metrics:
            for model_a, model_b in model_pairs:
                values_a = df[df['model'] == model_a][metric]
                values_b = df[df['model'] == model_b][metric]

                if len(values_a) > 0 and len(values_b) > 0:
                    stat, p_value = mannwhitneyu(values_a, values_b, alternative='two-sided')
                    results.append({
                        'metric': metric,
                        'model_a': model_a,
                        'model_b': model_b,
                        'u_statistic': stat,
                        'p_value': p_value,
                        'mean_model_a': values_a.mean(),
                        'mean_model_b': values_b.mean()
                    })

        results = pd.DataFrame(results)

        output_dir = Path(output_dir)
        results.to_csv(output_dir / "model_comparison.csv", index=False)
        print(f"Comparison results have been saved to {output_dir}/model_comparison.csv")

    def environmental_factors(self, env_csv):
        """
        env_csv: path to CSV containing environmental information about each video
        """
        env_df = pd.read_csv(env_csv)
        df, _ = self._summary(self.output_dir, save_as_csv=False)

        # merging datasets
        merged_df = pd.merge(df, env_df, on='video', how='left')

        env_factors = env_df.columns.tolist()
        env_factors.remove('video')

        # metrics to analyse
        metrics = ['pck_mean', 'mpjpe_mean', 'matched_frames']

        # removing columns with missing values across required columns
        cols_to_check = metrics + env_factors
        merged_df = merged_df.dropna(subset=cols_to_check)

        # Output file
        output_file = f'{self.output_dir}/regression_results.txt'

        # writing all outputs to output_file
        with open(output_file, 'w') as f:
            for model_name in merged_df['model'].unique():
                df_subset = merged_df[merged_df['model'] == model_name]

                for metric in metrics:
                    formula = f"{metric} ~ " + " + ".join([f"C({col})" for col in env_factors])
                    result = smf.ols(formula=formula, data=df_subset).fit()

                    f.write(f"\n{'='*80}\n")
                    f.write(f"Model: {model_name}, Metric: {metric}\n")
                    f.write(f"{'-'*80}\n")
                    f.write(result.summary().as_text())
                    f.write("\n\n\n")
        print(f"Regression results saved to: {output_file}")

    def visualisation(self):

        df, df_individual = self._summary(self.output_dir, save_as_csv=False)
        palette = sns.color_palette("colorblind", n_colors=df['model'].nunique())
        #violin plot for all PCK and MJPJE for each model
        def plot_all_distribution(data, metric, title, filter_note=""):
            plt.figure(figsize=(10, 6))
            plot_mean = 0

            # Set colour palette manually to keep consistency
            model_order = sorted(data['model'].unique())
            color_map = dict(zip(model_order, palette))

            for model in model_order:
                subset = data[data['model'] == model]
                color = color_map[model]

                # KDE Plot
                sns.kdeplot(subset[metric], fill=True, color=color, label=model, alpha=0.4)

                # Mean line
                mean_val = subset[metric].mean()
                plot_mean += mean_val
                plt.axvline(mean_val, linestyle='--', color=color)
                plt.text(mean_val, plt.ylim()[1]*0.9, f"Mean: {mean_val:.1f}",
                         rotation=90, va='top', ha='right', fontsize=18, color=color)

            plot_mean = (plot_mean/len(model_order))

            # Title
            full_title = f"{title} Distribution by Model" + (f" ({filter_note})" if filter_note else "")
            plt.title(full_title, fontsize=26)

            # Fully capitalised x-axis label
            plt.xlabel(metric.upper(), fontsize=22)
            plt.ylabel("Density", fontsize=22)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            if plot_mean/max(subset[metric]) < 0.5: # if the means mean is less than half of the max value, legend goes top right
                plt.legend(title="Model", loc="upper right", fontsize=20, title_fontsize=22)
            else:
                plt.legend(title="Model", loc="upper left", fontsize=20, title_fontsize=22)
            plt.tight_layout()
            plt.show()


        if self.pck_treshold and self.scale:
            plot_all_distribution(df_individual, "pck", f"PCK (< {self.pck_treshold *100}%)")
        elif self.pck_treshold and not self.scale:
            plot_all_distribution(df_individual, "pck", f"PCK (< {self.pck_treshold} pixels)")
        elif self.scale:
            plot_all_distribution(df_individual, "pck", f"PCK (< 50%)")
        else:
            plot_all_distribution(df_individual, "pck", "PCK (< 10 pixels)")
        plot_all_distribution(df_individual, "mpjpe", f"MPJPE")

        # Matched Frames violin plot
        plt.figure(figsize=(10, 6))
        model_order = sorted(df['model'].unique())
        model_palette = dict(zip(model_order, palette))
        ax = sns.violinplot(data=df, x="model", y="matched_frames", palette=model_palette, inner=None)
        
        # Get model names in the order they're plotted
        plotted_model_names = [tick.get_text() for tick in ax.get_xticklabels()]
        
        # Group stats by the order shown in the plot
        group_stats = df.groupby('model')['matched_frames'].agg(['mean', 'std'])
        group_stats = group_stats.reindex(plotted_model_names)
        
        # Get video counts in the same order
        video_counts = df['model'].value_counts().reindex(plotted_model_names)
        
        # Annotate each model
        for i, model in enumerate(plotted_model_names):
            mean = group_stats.loc[model, 'mean']
            std = group_stats.loc[model, 'std']
        
            # Get a slightly darker colour from the palette
            base_colour = model_palette[model]
            dark_colour = tuple(np.array(base_colour) * 0.4)
        
            # Vertical line from mean - 2*std to mean + 2*std
            ax.vlines(i, max(mean - 2 * std, 0), mean + 2 * std, color=dark_colour, linewidth=2)
        
            # Box from mean - std to mean + std
            box = patches.Rectangle(
                (i - 0.0125, max(mean - std, 0)), 0.025, 2 * std,
                linewidth=1.5,
                facecolor=dark_colour
            )
            ax.add_patch(box)
        
            # Horizontal mean line
            ax.hlines(mean, i - 0.0125, i + 0.0125, color="white", linewidth=2)
        
            # Mean label
            ax.text(i + 0.05, mean, f'{mean:.1f}', ha='left', va='center', fontsize=18, color=dark_colour)
        
            # Video count label under model name
            ax.text(i, -0.125 * df['matched_frames'].max(), f'n={video_counts[model]}',
                    ha='center', va='top', fontsize=16, color='black')
        
        # Plot aesthetics
        plt.title("Matched Frames per video", fontsize=26)
        plt.ylabel("Matched Frames", fontsize=22)
        plt.xlabel("Model", fontsize=22)
        plt.yticks(fontsize=22)
        plt.xticks(fontsize=22)
        
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.show()











        
