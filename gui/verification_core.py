"""Core functionality for model verification including worker threads."""

import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import logging
from collections import Counter
import torch
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger("Test-Annotation & KI-Modell Verifikation")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class OptimizeThresholdsWorker(QThread):
    """Worker thread for threshold optimization."""
    progress_updated = pyqtSignal(int)
    stage_updated = pyqtSignal(str)
    optimization_finished = pyqtSignal(dict)

    def __init__(self, model_path, image_list, step_size=5,
                 log_file=None, output_dir=None):
        super().__init__()
        self.model_path = model_path
        self.image_list = image_list
        self.step_size = step_size
        self.total_progress = 0
        self.log_file = log_file
        self.output_dir = output_dir
        
        # Set up detailed logging
        self.logger = logging.getLogger("threshold_optimization")
        self.logger.setLevel(logging.INFO)

        # Add file handler if log file is provided
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)

    def plot_accuracy_heatmap(self, history, outpath):
        """Generate heatmap visualization of accuracy results."""
        confs = [x[0]*100 for x in history]
        ious = [x[1]*100 for x in history]
        accs = [x[2] for x in history]

        unique_confs = sorted(list(set(confs)))
        unique_ious = sorted(list(set(ious)))
        heatmap = np.zeros((len(unique_ious), len(unique_confs)), dtype=float)
        
        for c, i, a in zip(confs, ious, accs):
            c_idx = unique_confs.index(c)
            i_idx = unique_ious.index(i)
            heatmap[i_idx, c_idx] = a
        
        plt.figure(figsize=(8, 6))
        plt.imshow(
            heatmap, 
            origin='lower',
            aspect='auto',
            cmap='RdYlGn',
            vmin=0, vmax=100,
            extent=[min(unique_confs), max(unique_confs),
                    min(unique_ious), max(unique_ious)]
        )
        plt.colorbar(label='Accuracy (%)')
        plt.xlabel("Confidence (%)")
        plt.ylabel("IoU (%)")
        plt.title("Accuracy-Heatmap in Abhängigkeit von Conf & IoU")
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()

    def plot_accuracy_surface(self, history, outpath):
        """Generate 3D surface plot of accuracy results."""
        confs = [x[0]*100 for x in history]
        ious = [x[1]*100 for x in history]
        accs = [x[2] for x in history]

        unique_confs = sorted(list(set(confs)))
        unique_ious = sorted(list(set(ious)))
        z_matrix = np.zeros((len(unique_ious), len(unique_confs)), dtype=float)

        for c, i, a in zip(confs, ious, accs):
            c_idx = unique_confs.index(c)
            i_idx = unique_ious.index(i)
            z_matrix[i_idx, c_idx] = a
        
        X, Y = np.meshgrid(unique_confs, unique_ious)
        Z = z_matrix

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(
            X, Y, Z,
            cmap='viridis',
            edgecolor='none',
            alpha=0.9
        )
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Accuracy (%)')
        ax.set_xlabel("Confidence (%)")
        ax.set_ylabel("IoU (%)")
        ax.set_zlabel("Accuracy (%)")
        ax.set_title("3D-Surface: Accuracy in Abhängigkeit von Conf & IoU")
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()

    def search_grid(self, conf_range, iou_range, step, progress_weight=1.0, base_progress=0, model=None, stage_label=""):
        """Search for optimal thresholds in a given range."""
        if model is None:
            raise ValueError("Model must be provided")
            
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Starting {stage_label} Search")
        self.logger.info(f"Configuration:")
        self.logger.info(f"- Confidence range: {conf_range}")
        self.logger.info(f"- IoU range: {iou_range}")
        self.logger.info(f"- Step size: {step}")
        self.logger.info(f"{'='*50}\n")
            
        start_conf = max(1, min(100, float(conf_range[0])))
        end_conf = max(1, min(100, float(conf_range[1])))
        start_iou = max(1, min(100, float(iou_range[0])))
        end_iou = max(1, min(100, float(iou_range[1])))
        step = float(step)
        
        best_result = {
            'conf': 0.25,
            'iou': 0.45,
            'accuracy': 0.0
        }
        
        conf_steps = max(1, int((end_conf - start_conf) / step) + 1)
        iou_steps = max(1, int((end_iou - start_iou) / step) + 1)
        total_combinations = conf_steps * iou_steps
        current_combination = 0
        
        batch_size = min(16, len(self.image_list))
        
        conf_range = np.clip(np.arange(start_conf, end_conf + step/2, step), 1, 100)
        iou_range = np.clip(np.arange(start_iou, end_iou + step/2, step), 1, 100)
        
        self.logger.info(f"Testing {len(conf_range) * len(iou_range)} combinations...")
        search_history = []
        total_images = len(self.image_list)

        for conf in conf_range:
            for iou in iou_range:
                good_count = 0
                total_images = len(self.image_list)

                conf_threshold = np.clip(conf / 100.0, 0.01, 0.99)
                iou_threshold = np.clip(iou / 100.0, 0.01, 0.99)
                
                self.logger.info(f"\nTesting combination:")
                self.logger.info(f"Confidence: {conf_threshold:.3f}, IoU: {iou_threshold:.3f}")
                
                for i in range(0, total_images, batch_size):
                    batch = self.image_list[i:i+batch_size]
                    results = model.predict(
                        source=batch,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        show=False,
                        verbose=False,
                        device='cuda' if torch.cuda.is_available() else 'cpu'
                    )
                    
                    for img_path, result in zip(batch, results):
                        gt_counter = Counter()
                        annot_file = os.path.splitext(img_path)[0] + ".txt"
                        if os.path.exists(annot_file):
                            with open(annot_file, 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        cls = int(float(parts[0]))
                                        gt_counter[cls] += 1
                        
                        pred_counter = Counter()
                        if hasattr(result, "boxes") and result.boxes is not None:
                            for box in result.boxes:
                                cls_pred = int(box.cls[0].cpu().numpy())
                                pred_counter[cls_pred] += 1
                        
                        if gt_counter == pred_counter:
                            good_count += 1
                
                accuracy = (good_count / total_images) * 100

                search_history.append((
                    float(conf_threshold),
                    float(iou_threshold),
                    float(accuracy),
                    stage_label
                ))
                
                self.logger.info(f"Results for conf={conf_threshold:.3f}, iou={iou_threshold:.3f}:")
                self.logger.info(f"- Accuracy: {accuracy:.2f}%")
                self.logger.info(f"- Correctly annotated: {good_count}/{total_images}")
                if accuracy > best_result['accuracy']:
                    self.logger.info(f"=> New best result! Previous best: {best_result['accuracy']:.2f}%")
                self.logger.info("-" * 30)

                if accuracy > best_result['accuracy']:
                    best_result = {
                            'conf': conf_threshold,
                            'iou': iou_threshold,
                            'accuracy': accuracy
                        }
                                              
                current_combination += 1
                progress = base_progress + int((current_combination / total_combinations) * 100 * progress_weight)
                self.progress_updated.emit(progress)
                
                # Log progress
                if current_combination % 10 == 0:
                    self.logger.info(
                        f"Progress: {current_combination}/{total_combinations} combinations tested "
                        f"({(current_combination/total_combinations*100):.1f}%)"
                    )
                if best_result['accuracy'] >= 99:
                    self.progress_updated.emit(progress)
                    return best_result, search_history
                
                if self.isInterruptionRequested():
                    return best_result, search_history
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"{stage_label} Search Complete")
        self.logger.info(f"Best result:")
        self.logger.info(f"- Confidence: {best_result['conf']:.3f}")
        self.logger.info(f"- IoU: {best_result['iou']:.3f}")
        self.logger.info(f"- Accuracy: {best_result['accuracy']:.2f}%")
        self.logger.info(f"{'='*50}\n")
        return best_result, search_history

    def run(self):
        """Execute threshold optimization."""
        try:
            model = YOLO(self.model_path)
            self.model = model

            # Determine output directory
            if not self.output_dir:
                model_dir = os.path.dirname(os.path.abspath(self.model_path))
                parent_dir = os.path.dirname(model_dir)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.output_dir = os.path.join(parent_dir, f"verification_{timestamp}")
            os.makedirs(self.output_dir, exist_ok=True)

            # Set up log file
            if not self.log_file:
                self.log_file = os.path.join(self.output_dir, "threshold_optimization.log")
            self.logger.addHandler(logging.FileHandler(self.log_file))

            self.plot_dir = os.path.join(self.output_dir, "optimization_plots")
            os.makedirs(self.plot_dir, exist_ok=True)

            best_result = {
                'conf': 0.25,
                'iou': 0.45,
                'accuracy': 0.0
            }
            
            self.logger.info("\nStarting Threshold Optimization")
            self.logger.info(f"Model: {self.model_path}")
            self.logger.info(f"Images: {len(self.image_list)}")
            self.logger.info(f"Step size: {self.step_size}%")
            self.logger.info(f"Plot directory: {self.plot_dir}\n")

            # Stage 1: Coarse search
            self.stage_updated.emit("Stufe 1: Grobes Tuning (spart Zeit)")
            coarse_step = 20
            coarse_best, coarse_history = self.search_grid(
                conf_range=(1, 100),
                iou_range=(1, 100),
                step=coarse_step,
                progress_weight=0.3,
                base_progress=0,
                model=model,
                stage_label="Grob"
            )

            # Stage 2: Medium search
            self.stage_updated.emit("Stufe 2: 10%-Schritte (grenzt den Optimalbereich ein)")
            conf_best = coarse_best['conf'] * 100
            iou_best = coarse_best['iou'] * 100
            medium_step = 10

            conf_min = max(1, conf_best - coarse_step)
            conf_max = min(100, conf_best + coarse_step)
            iou_min = max(1, iou_best - coarse_step)
            iou_max = min(100, iou_best + coarse_step)

            medium_best, medium_history = self.search_grid(
                conf_range=(conf_min, conf_max),
                iou_range=(iou_min, iou_max),
                step=medium_step,
                progress_weight=0.3,
                base_progress=43,
                model=model,
                stage_label="Mittel"
            )

            # Stage 3: Fine search
            self.stage_updated.emit("Stufe 3: Fein-Tuning, um den Optimalbereich präzise zu bestimmen")
            conf_best = medium_best['conf'] * 100
            iou_best = medium_best['iou'] * 100

            conf_min = max(1, conf_best - medium_step)
            conf_max = min(100, conf_best + medium_step)
            iou_min = max(1, iou_best - medium_step)
            iou_max = min(100, iou_best + medium_step)

            best_result, fine_history = self.search_grid(
                conf_range=(conf_min, conf_max),
                iou_range=(iou_min, iou_max),
                step=self.step_size,
                progress_weight=0.4,
                base_progress=73,
                model=model,
                stage_label="Fein"
            )

            all_history = coarse_history + medium_history + fine_history            

            self.progress_updated.emit(98)
            self.stage_updated.emit("Stufe 4: Tuning-Plots werden erstellt")        

            heatmap_path = os.path.join(self.plot_dir, "accuracy_heatmap.png")
            self.plot_accuracy_heatmap(all_history, heatmap_path)
            self.progress_updated.emit(99)

            surface_path = os.path.join(self.plot_dir, "accuracy_surface.png")
            self.plot_accuracy_surface(all_history, surface_path)
            self.progress_updated.emit(100)                                    
                
            # Log final results
            self.logger.info("\nOptimization Complete!")
            self.logger.info("Final Results:")
            self.logger.info(f"- Best Confidence: {best_result['conf']:.3f}")
            self.logger.info(f"- Best IoU: {best_result['iou']:.3f}")
            self.logger.info(f"- Best Accuracy: {best_result['accuracy']:.2f}%")
            self.logger.info(f"- Log file: {self.log_file}")
            self.logger.info(f"- Plot directory: {self.plot_dir}")
            self.optimization_finished.emit(best_result)
            
        except Exception as e:
            logger.error(f"Error during threshold optimization: {e}")
            self.optimization_finished.emit({})

class AnnotationWorker(QThread):
    """Worker thread for image annotation and verification."""
    mosaic_updated = pyqtSignal(QPixmap)
    progress_updated = pyqtSignal(int)
    summary_signal = pyqtSignal(str, float, str)  # text, percentage, misannotated_dir
    finished = pyqtSignal()

    def __init__(self, model_path, image_list, output_dir, threshold, iou_threshold, tile_size=200):
        super().__init__()
        self.model_path = model_path
        self.image_list = image_list
        self.output_dir = output_dir
        self.misannotated_dir = ""
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.tile_size = tile_size
        self.mosaic_history = []
        self.current_mosaic_index = -1

    def run(self):
        """Execute image annotation and verification."""
        model = YOLO(self.model_path)
        total_images = len(self.image_list)

        if not self.output_dir:
            model_dir = os.path.dirname(os.path.abspath(self.model_path))
            parent_dir = os.path.dirname(model_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(parent_dir, f"verification_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.misannotated_dir = os.path.join(self.output_dir, "misannotated")
        os.makedirs(self.misannotated_dir, exist_ok=True)
        
        current_index = 0
        batch_size = 9
        bad_count = 0
        good_count = 0
        false_index = 1

        while current_index < total_images:
            batch = self.image_list[current_index:current_index+batch_size]
            if len(batch) < batch_size:
                batch += [""] * (batch_size - len(batch))
            valid_images = [img for img in batch if img]
            results = None
            if valid_images:
                results = model.predict(source=valid_images, conf=self.threshold, iou=self.iou_threshold, show=False, verbose=False)
            ts = self.tile_size
            mosaic = np.zeros((3*ts, 3*ts, 3), dtype=np.uint8)
            
            for i, img_path in enumerate(batch):
                if not img_path:
                    continue

                orig_img = cv2.imread(img_path)
                if orig_img is None:
                    continue

                annotated_img = orig_img.copy()
                h, w = annotated_img.shape[:2]
                
                gt_counter = Counter()
                base = os.path.splitext(img_path)[0]
                annot_file = base + ".txt"
                if os.path.exists(annot_file):
                    with open(annot_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls = int(float(parts[0]))
                                gt_counter[cls] += 1
                                x_center = float(parts[1]) * w
                                y_center = float(parts[2]) * h
                                bw = float(parts[3]) * w
                                bh = float(parts[4]) * h
                                x1 = int(x_center - bw/2)
                                y1 = int(y_center - bh/2)
                                x2 = int(x_center + bw/2)
                                y2 = int(y_center + bh/2)
                                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated_img, str(cls), (x1, max(y1-5, 0)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                
                pred_counter = Counter()
                if results is not None and img_path in valid_images:
                    idx = valid_images.index(img_path)
                    r = results[idx]
                    if hasattr(r, "boxes") and r.boxes is not None:
                        for box in r.boxes:
                            coords = box.xyxy[0].cpu().numpy().astype(int)
                            cls_pred = int(box.cls[0].cpu().numpy())
                            pred_counter[cls_pred] += 1
                            cv2.rectangle(annotated_img, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), 2)
                            cv2.putText(annotated_img, str(cls_pred), (coords[0], max(coords[1]-5,0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                
                if gt_counter != pred_counter:
                    bad_flag = True
                    bad_count += 1
                    missing = gt_counter - pred_counter
                    extra = pred_counter - gt_counter
                    ext = os.path.splitext(img_path)[1]
                    new_filename = f"false_img{false_index}{ext}"
                    false_index += 1
                    dest_file = os.path.join(self.misannotated_dir, new_filename)
                    cv2.imwrite(dest_file, annotated_img)
                    log_line = (f"{new_filename}: Erwartet: {dict(gt_counter)}, Vorhergesagt: {dict(pred_counter)}. "
                                f"Fehlend: {dict(missing)}, Überschuss: {dict(extra)}")
                    logger.info("Falsch annotiert: " + log_line)
                else:
                    bad_flag = False
                    good_count += 1
                
                display_img = cv2.resize(annotated_img, (ts, ts))
                if bad_flag:
                    overlay = display_img.copy()
                    overlay[:] = (0, 0, 255)
                    alpha = 0.25
                    cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)
                
                row = i // 3
                col = i % 3
                mosaic[row*ts:(row+1)*ts, col*ts:(col+1)*ts] = display_img
            
            rgb_mosaic = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_mosaic.shape
            bytesPerLine = 3 * width
            qImg = QImage(rgb_mosaic.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.mosaic_updated.emit(pixmap)
            self.mosaic_history.append(pixmap)
            self.current_mosaic_index = len(self.mosaic_history) - 1
            progress = int(min(100, (current_index + batch_size) / total_images * 100))
            self.progress_updated.emit(progress)
            current_index += batch_size
        
        summary = (f"Live Annotation abgeschlossen.\n"
                  f"Gesamtbilder: {total_images}\n"
                  f"Korrekt annotiert: {good_count}\n"
                  f"Falsch annotiert: {bad_count}\n"
                  f"Falsch annotierte Bilder im Ordner: {self.misannotated_dir}")
        
        correct_percentage = (good_count / total_images) * 100

        self.summary_signal.emit(summary, correct_percentage, self.misannotated_dir)
        self.finished.emit()