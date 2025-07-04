import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import glob
import json
from pathlib import Path
import shutil

class YOLOQualityChecker:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Annotation Quality Checker")
        self.root.geometry("1400x900")
        
        # Datenstrukturen
        self.image_files = []
        self.current_index = 0
        self.dataset_path = ""
        self.quality_issues = {}
        self.deleted_files = []
        
        # Qualitätskriterien (anpassbar)
        self.quality_thresholds = {
            'min_box_size': 0.01,  # Mindestgröße als Anteil der Bildgröße
            'edge_threshold': 0.02,  # Abstand zum Bildrand
            'min_width_height': 0.005,  # Minimale Breite/Höhe
            'aspect_ratio_extreme': 20  # Extreme Seitenverhältnisse
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Hauptframe
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Obere Steuerungsleiste
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(control_frame, text="Dataset Ordner wählen", 
                  command=self.select_dataset_folder).pack(side='left', padx=(0, 10))
        
        # Qualitätsfilter
        ttk.Label(control_frame, text="Filter:").pack(side='left', padx=(20, 5))
        self.filter_var = tk.StringVar(value="alle")
        filter_combo = ttk.Combobox(control_frame, textvariable=self.filter_var, 
                                   values=["alle", "kritisch", "ok"], state="readonly", width=10)
        filter_combo.pack(side='left', padx=(0, 10))
        filter_combo.bind('<<ComboboxSelected>>', self.apply_filter)
        
        # Statistik Label
        self.stats_label = ttk.Label(control_frame, text="Keine Daten geladen")
        self.stats_label.pack(side='right')
        
        # Hauptcontainer
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill='both', expand=True)
        
        # Linke Seite - Galerie
        gallery_frame = ttk.LabelFrame(content_frame, text="Galerie", padding=10)
        gallery_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Scrollbare Galerie
        self.setup_gallery(gallery_frame)
        
        # Rechte Seite - Detailansicht
        detail_frame = ttk.LabelFrame(content_frame, text="Detailansicht", padding=10)
        detail_frame.pack(side='right', fill='both', expand=True)
        
        self.setup_detail_view(detail_frame)
        
    def setup_gallery(self, parent):
        # Scrollbarer Container für Galerie
        canvas = tk.Canvas(parent, bg='white')
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.gallery_frame = ttk.Frame(canvas)
        
        self.gallery_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.gallery_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.gallery_canvas = canvas
        
    def setup_detail_view(self, parent):
        # Bildanzeige
        self.image_label = ttk.Label(parent, text="Kein Bild gewählt")
        self.image_label.pack(fill='both', expand=True)
        
        # Navigation
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(nav_frame, text="◀ Vorheriges", 
                  command=self.previous_image).pack(side='left')
        
        self.image_info_label = ttk.Label(nav_frame, text="")
        self.image_info_label.pack(side='left', expand=True)
        
        ttk.Button(nav_frame, text="Nächstes ▶", 
                  command=self.next_image).pack(side='right')
        
        # Qualitätsinformationen
        quality_frame = ttk.LabelFrame(parent, text="Qualitätsprobleme", padding=5)
        quality_frame.pack(fill='x', pady=(10, 0))
        
        self.quality_text = tk.Text(quality_frame, height=4, wrap='word')
        self.quality_text.pack(fill='x')
        
        # Aktionen
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(action_frame, text="🗑️ Löschen", 
                  command=self.delete_current_pair,
                  style='Danger.TButton').pack(side='left')
        
        ttk.Button(action_frame, text="📁 Im Explorer öffnen", 
                  command=self.open_in_explorer).pack(side='right')
        
    def select_dataset_folder(self):
        folder = filedialog.askdirectory(title="YOLO Dataset Ordner wählen")
        if folder:
            self.dataset_path = folder
            self.load_dataset()
            
    def load_dataset(self):
        """Lädt das YOLO-Dataset und analysiert die Qualität"""
        try:
            # Finde alle Bilder
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            self.image_files = []
            
            for ext in image_extensions:
                self.image_files.extend(glob.glob(os.path.join(self.dataset_path, ext)))
                self.image_files.extend(glob.glob(os.path.join(self.dataset_path, ext.upper())))
            
            if not self.image_files:
                messagebox.showerror("Fehler", "Keine Bilder im gewählten Ordner gefunden!")
                return
            
            self.image_files.sort()
            self.current_index = 0
            self.quality_issues = {}
            
            # Qualitätsanalyse
            self.analyze_quality()
            
            # UI aktualisieren
            self.update_gallery()
            self.update_detail_view()
            self.update_stats()
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Laden des Datasets: {str(e)}")
    
    def analyze_quality(self):
        """Analysiert die Qualität aller Annotationen"""
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Analysiere Qualität...")
        progress_window.geometry("300x100")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        progress_label = ttk.Label(progress_window, text="Analysiere Annotationen...")
        progress_label.pack(pady=10)
        
        progress_bar = ttk.Progressbar(progress_window, length=250, mode='determinate')
        progress_bar.pack(pady=10)
        progress_bar['maximum'] = len(self.image_files)
        
        for i, img_path in enumerate(self.image_files):
            progress_bar['value'] = i
            progress_window.update()
            
            issues = self.check_image_quality(img_path)
            if issues:
                self.quality_issues[img_path] = issues
        
        progress_window.destroy()
    
    def check_image_quality(self, img_path):
        """Prüft die Qualität einer einzelnen Annotation"""
        issues = []
        
        # Entsprechende Label-Datei finden
        label_path = os.path.splitext(img_path)[0] + '.txt'
        if not os.path.exists(label_path):
            issues.append("Keine Label-Datei gefunden")
            return issues
        
        try:
            # Bildgröße laden
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            # Labels lesen
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                issues.append("Label-Datei ist leer")
                return issues
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) < 5:
                        issues.append(f"Zeile {line_num}: Unvollständige Annotation")
                        continue
                    
                    class_id, x_center, y_center, width, height = map(float, parts[:5])
                    
                    # Prüfe Wertebereiche
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                           0 <= width <= 1 and 0 <= height <= 1):
                        issues.append(f"Zeile {line_num}: Koordinaten außerhalb des Bereichs [0,1]")
                    
                    # Prüfe Mindestgröße
                    box_area = width * height
                    if box_area < self.quality_thresholds['min_box_size']:
                        issues.append(f"Zeile {line_num}: Bounding Box zu klein (Fläche: {box_area:.4f})")
                    
                    # Prüfe Mindestbreite/-höhe
                    if width < self.quality_thresholds['min_width_height']:
                        issues.append(f"Zeile {line_num}: Box zu schmal (Breite: {width:.4f})")
                    if height < self.quality_thresholds['min_width_height']:
                        issues.append(f"Zeile {line_num}: Box zu niedrig (Höhe: {height:.4f})")
                    
                    # Prüfe Seitenverhältnis
                    aspect_ratio = max(width/height, height/width) if min(width, height) > 0 else float('inf')
                    if aspect_ratio > self.quality_thresholds['aspect_ratio_extreme']:
                        issues.append(f"Zeile {line_num}: Extremes Seitenverhältnis ({aspect_ratio:.1f}:1)")
                    
                    # Prüfe Bildrandnähe
                    x_min = x_center - width/2
                    x_max = x_center + width/2
                    y_min = y_center - height/2
                    y_max = y_center + height/2
                    
                    threshold = self.quality_thresholds['edge_threshold']
                    if (x_min < threshold or x_max > 1-threshold or 
                        y_min < threshold or y_max > 1-threshold):
                        issues.append(f"Zeile {line_num}: Box berührt Bildrand")
                    
                    # Prüfe Abschneidung
                    if x_min < 0 or x_max > 1 or y_min < 0 or y_max > 1:
                        issues.append(f"Zeile {line_num}: Box ist abgeschnitten")
                
                except ValueError:
                    issues.append(f"Zeile {line_num}: Ungültige Zahlenformate")
        
        except Exception as e:
            issues.append(f"Fehler beim Lesen der Dateien: {str(e)}")
        
        return issues
    
    def update_gallery(self):
        """Aktualisiert die Galerie-Ansicht"""
        # Lösche alte Thumbnails
        for widget in self.gallery_frame.winfo_children():
            widget.destroy()
        
        # Filtere Bilder nach aktuellem Filter
        filtered_images = self.get_filtered_images()
        
        # Erstelle Thumbnails
        cols = 4
        for i, img_path in enumerate(filtered_images):
            row = i // cols
            col = i % cols
            
            try:
                # Thumbnail erstellen
                with Image.open(img_path) as img:
                    img.thumbnail((150, 150))
                    
                    # Qualitätsstatus visualisieren
                    if img_path in self.quality_issues:
                        # Roten Rahmen für kritische Bilder
                        bordered_img = Image.new('RGB', (img.width + 4, img.height + 4), 'red')
                        bordered_img.paste(img, (2, 2))
                        photo = ImageTk.PhotoImage(bordered_img)
                    else:
                        # Grünen Rahmen für OK-Bilder
                        bordered_img = Image.new('RGB', (img.width + 4, img.height + 4), 'green')
                        bordered_img.paste(img, (2, 2))
                        photo = ImageTk.PhotoImage(bordered_img)
                
                # Thumbnail Button
                btn = tk.Button(self.gallery_frame, image=photo, 
                              command=lambda p=img_path: self.select_image(p))
                btn.image = photo  # Referenz behalten
                btn.grid(row=row, column=col, padx=2, pady=2)
                
                # Dateiname Label
                filename = os.path.basename(img_path)
                if len(filename) > 20:
                    filename = filename[:17] + "..."
                
                label = ttk.Label(self.gallery_frame, text=filename, font=('Arial', 8))
                label.grid(row=row*2+1, column=col, padx=2, pady=(0, 5))
                
            except Exception as e:
                print(f"Fehler beim Erstellen des Thumbnails für {img_path}: {e}")
    
    def get_filtered_images(self):
        """Gibt gefilterte Bildliste zurück"""
        if self.filter_var.get() == "kritisch":
            return [img for img in self.image_files if img in self.quality_issues]
        elif self.filter_var.get() == "ok":
            return [img for img in self.image_files if img not in self.quality_issues]
        else:
            return self.image_files
    
    def select_image(self, img_path):
        """Wählt ein Bild aus der Galerie"""
        try:
            self.current_index = self.image_files.index(img_path)
            self.update_detail_view()
        except ValueError:
            pass
    
    def update_detail_view(self):
        """Aktualisiert die Detailansicht"""
        if not self.image_files:
            return
        
        img_path = self.image_files[self.current_index]
        
        try:
            # Bild mit Annotationen laden
            annotated_img = self.load_image_with_annotations(img_path)
            
            # Bild für Anzeige skalieren
            display_img = annotated_img.copy()
            display_img.thumbnail((600, 600))
            photo = ImageTk.PhotoImage(display_img)
            
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
            
            # Bildinformationen
            filename = os.path.basename(img_path)
            info = f"{self.current_index + 1}/{len(self.image_files)}: {filename}"
            self.image_info_label.configure(text=info)
            
            # Qualitätsinformationen
            self.quality_text.delete(1.0, tk.END)
            if img_path in self.quality_issues:
                issues_text = "\n".join(self.quality_issues[img_path])
                self.quality_text.insert(1.0, issues_text)
                self.quality_text.configure(bg='#ffeeee')
            else:
                self.quality_text.insert(1.0, "✓ Keine Qualitätsprobleme erkannt")
                self.quality_text.configure(bg='#eeffee')
            
        except Exception as e:
            self.image_label.configure(image="", text=f"Fehler beim Laden: {str(e)}")
    
    def load_image_with_annotations(self, img_path):
        """Lädt ein Bild und zeichnet die Annotationen ein"""
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # Label-Datei lesen
            label_path = os.path.splitext(img_path)[0] + '.txt'
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                img_width, img_height = img.size
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id, x_center, y_center, width, height = map(float, parts[:5])
                            
                            # Koordinaten in Pixel umrechnen
                            x_min = int((x_center - width/2) * img_width)
                            y_min = int((y_center - height/2) * img_height)
                            x_max = int((x_center + width/2) * img_width)
                            y_max = int((y_center + height/2) * img_height)
                            
                            # Farbe je nach Qualität
                            color = 'red' if img_path in self.quality_issues else 'green'
                            
                            # Bounding Box zeichnen
                            draw.rectangle([x_min, y_min, x_max, y_max], 
                                         outline=color, width=3)
                            
                            # Klassen-ID anzeigen
                            draw.text((x_min, y_min-20), f"Class: {int(class_id)}", 
                                    fill=color)
                    
                    except ValueError:
                        continue
            
            return img
    
    def previous_image(self):
        """Zeigt das vorherige Bild"""
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.update_detail_view()
    
    def next_image(self):
        """Zeigt das nächste Bild"""
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.update_detail_view()
    
    def delete_current_pair(self):
        """Löscht das aktuelle Bild-Label-Paar"""
        if not self.image_files:
            return
        
        img_path = self.image_files[self.current_index]
        label_path = os.path.splitext(img_path)[0] + '.txt'
        
        # Bestätigung
        filename = os.path.basename(img_path)
        if not messagebox.askyesno("Löschen bestätigen", 
                                  f"Möchten Sie '{filename}' und die zugehörige Label-Datei wirklich löschen?"):
            return
        
        try:
            # Dateien löschen
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(label_path):
                os.remove(label_path)
            
            # Aus Listen entfernen
            self.deleted_files.append(img_path)
            self.image_files.remove(img_path)
            if img_path in self.quality_issues:
                del self.quality_issues[img_path]
            
            # Index anpassen
            if self.current_index >= len(self.image_files):
                self.current_index = len(self.image_files) - 1
            
            # UI aktualisieren
            self.update_gallery()
            self.update_detail_view()
            self.update_stats()
            
            messagebox.showinfo("Gelöscht", f"'{filename}' wurde gelöscht.")
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Löschen: {str(e)}")
    
    def open_in_explorer(self):
        """Öffnet das aktuelle Bild im Datei-Explorer"""
        if not self.image_files:
            return
        
        img_path = self.image_files[self.current_index]
        
        try:
            import subprocess
            import platform
            
            if platform.system() == "Windows":
                subprocess.run(["explorer", "/select,", img_path])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", "-R", img_path])
            else:  # Linux
                subprocess.run(["xdg-open", os.path.dirname(img_path)])
        
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte Explorer nicht öffnen: {str(e)}")
    
    def apply_filter(self, event=None):
        """Wendet den gewählten Filter an"""
        self.update_gallery()
    
    def update_stats(self):
        """Aktualisiert die Statistik-Anzeige"""
        if not self.image_files:
            self.stats_label.configure(text="Keine Daten geladen")
            return
        
        total = len(self.image_files)
        critical = len(self.quality_issues)
        ok = total - critical
        
        stats_text = f"Gesamt: {total} | Kritisch: {critical} | OK: {ok}"
        if self.deleted_files:
            stats_text += f" | Gelöscht: {len(self.deleted_files)}"
        
        self.stats_label.configure(text=stats_text)

def main():
    root = tk.Tk()
    
    # Stil für Danger-Button
    style = ttk.Style()
    style.configure('Danger.TButton', foreground='red')
    
    app = YOLOQualityChecker(root)
    
    # Keyboard shortcuts
    root.bind('<Left>', lambda e: app.previous_image())
    root.bind('<Right>', lambda e: app.next_image())
    root.bind('<Delete>', lambda e: app.delete_current_pair())
    
    root.mainloop()

if __name__ == "__main__":
    main()