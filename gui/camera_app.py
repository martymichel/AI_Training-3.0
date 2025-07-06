import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import time
import urllib.request
from PIL import Image, ImageTk
import io
import json
import yaml
import numpy as np
import cv2
from pathlib import Path
import os
import subprocess
import sys

# YOLO Detection imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics YOLO nicht verfügbar. Objekterkennung deaktiviert.")

# Import IDS NXT API components
from .src.nxt_rest_connection import NXTRestConnection
from .src.nxt_camera_handler_base import NXTCameraHandlerBase
from .src.nxt_streaming_handler import NXTStreamingHandler
from .src.nxt_config import NXTConfig
from project_manager import ProjectManager

class IDSNXTCameraApp:
    def __init__(self, root, settings_dir="."):
        self.root = root
        self.settings_dir = Path(settings_dir)
        # Project specific manager for persistent settings
        self.project_manager = ProjectManager(settings_dir)
        self.root = root
        self.root.title("IDS NXT Kamera Live-Streaming")
        # Windows Fenster maximieren
        self.root.state('zoomed')
        
        self.root.configure(bg='#f0f0f0')
        
        # Kamera-Verbindungsparameter
        self.rest_connection = None
        self.camera_handler = None
        self.streaming_handler = None
        
        # Einstellungen-Management
        self.settings_file = self.settings_dir / "camera_settings.json"
        self.settings = self.load_all_settings()
        
        # Streaming-Variablen
        self.streaming_active = False
        self.current_stream_url = None
        self.stream_thread = None
        self.streaming_available = False
        
        # System Monitoring
        self.monitor_active = False
        self.monitor_thread = None
        self.system_data = {}
        
        # Performance-Optimierungen für Rio Live-Stream
        self.image_buffer = None
        self.last_image_time = 0
        self.fps_counter = 0
        self.fps_last_time = time.time()
        self.current_live_fps = 0  # Echte Live-Stream FPS
        self.adaptive_quality = 70
        self.target_fps = 15  # Ziel-FPS für Live-Bilder
        
        # YOLO Detection
        self.yolo_model = None
        self.class_names = {}
        self.detection_enabled = False
        self.detection_settings_file = self.settings_dir / "detection_settings.json"
        self.detection_settings = self.load_detection_settings()
        self.motion_threshold = 110
        self.prev_gray = None
        self.is_static = False
        
        # GUI erstellen
        self.create_gui()
        
        # Auto-Save Setup
        self.setup_settings_auto_save()
        
        # Automatische Verbindung wenn Daten vorhanden
        self.load_settings_to_gui()
        
        # Automatische Verbindung wenn Daten vorhanden
        if self.settings.get('connection', {}).get('ip'):
            self.auto_connect()
    
    def load_all_settings(self):
        """Lädt Einstellungen aus Datei und Projekt-Manager"""
        try:
            settings = {}
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)

            default_settings = {
                "connection": {"ip": "192.168.1.99", "user": "admin", "password": "Flex"},
                "streaming": {"stream_type": "stream1", "fps": 15, "quality": 70},
                "camera": {"exposure_time": 10000, "gain": 0.0, "flip_horizontal": False, "flip_vertical": False},
                "detection": {"model_path": "", "yaml_path": "", "motion_threshold": 110, "iou_threshold": 0.45, "class_thresholds": {}, "enabled": False},
            }

            for section, values in default_settings.items():
                if section not in settings:
                    settings[section] = {}
                for key, val in values.items():
                    settings[section].setdefault(key, val)

            pm_cam = self.project_manager.get_camera_settings()
            for key, val in pm_cam.items():
                settings["camera"][key] = val

            pm_live = self.project_manager.get_live_detection_settings()
            settings["connection"].update(pm_live.get("connection", {}))
            for k in ["stream_type", "fps", "quality"]:
                if k in pm_live:
                    settings["streaming"][k] = pm_live[k]
            for k in ["exposure_time", "gain", "flip_horizontal", "flip_vertical"]:
                if k in pm_live:
                    settings["camera"][k] = pm_live[k]

            if pm_live.get("model_path"):
                settings["detection"]["model_path"] = pm_live["model_path"]
            if pm_live.get("yaml_path"):
                settings["detection"]["yaml_path"] = pm_live["yaml_path"]
            for k in ["motion_threshold", "iou_threshold", "class_thresholds", "detection_enabled", "enabled"]:
                if k in pm_live:
                    target_key = "enabled" if k in ["detection_enabled", "enabled"] else k
                    settings["detection"][target_key] = pm_live[k]

            return settings

        except Exception as e:
            print(f"Fehler beim Laden der Einstellungen: {e}")
            return default_settings
    
    def save_all_settings(self):
        """Speichert alle aktuellen Einstellungen in JSON-Datei"""
        try:
            # Update settings from GUI
            self.settings['connection']['ip'] = self.ip_var.get()
            self.settings['connection']['user'] = self.user_var.get()
            self.settings['connection']['password'] = self.password_var.get()
            
            self.settings['streaming']['stream_type'] = self.stream_var.get()
            self.settings['streaming']['fps'] = self.fps_var.get()
            self.settings['streaming']['quality'] = self.quality_var.get()
            
            self.settings['camera']['exposure_time'] = self.exposure_var.get()
            self.settings['camera']['gain'] = self.gain_var.get()
            self.settings['camera']['flip_horizontal'] = self.flip_h_var.get()
            self.settings['camera']['flip_vertical'] = self.flip_v_var.get()

            # Detection settings are updated by their respective methods

            live_settings = {
                'connection': self.settings['connection'],
                'stream_type': self.settings['streaming']['stream_type'],
                'fps': self.settings['streaming']['fps'],
                'quality': self.settings['streaming']['quality'],
                'exposure_time': self.settings['camera']['exposure_time'],
                'gain': self.settings['camera']['gain'],
                'flip_horizontal': self.settings['camera']['flip_horizontal'],
                'flip_vertical': self.settings['camera']['flip_vertical'],
            }

            self.project_manager.update_camera_settings(self.settings['camera'])
            self.project_manager.update_live_detection_settings(live_settings)

            # Write to file
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"Fehler beim Speichern der Einstellungen: {e}")
    
    def load_settings_to_gui(self):
        """Lädt gespeicherte Einstellungen in die GUI"""
        try:
            # Connection settings
            conn = self.settings.get('connection', {})
            self.ip_var.set(conn.get('ip', '192.168.1.99'))
            self.user_var.set(conn.get('user', 'admin'))
            self.password_var.set(conn.get('password', 'Flex'))
            
            # Streaming settings
            stream = self.settings.get('streaming', {})
            self.stream_var.set(stream.get('stream_type', 'stream1'))
            self.fps_var.set(stream.get('fps', 15))
            self.quality_var.set(stream.get('quality', 70))
            
            # Camera settings
            camera = self.settings.get('camera', {})
            self.exposure_var.set(camera.get('exposure_time', 10000))
            self.gain_var.set(camera.get('gain', 0.0))
            self.flip_h_var.set(camera.get('flip_horizontal', False))
            self.flip_v_var.set(camera.get('flip_vertical', False))
            
            # Update labels
            self.fps_label.configure(text=f"{self.fps_var.get()} FPS")
            self.quality_label.configure(text=f"{self.quality_var.get()}%")
            self.exposure_label.configure(text=f"{self.exposure_var.get()} µs")
            self.gain_label.configure(text=f"{self.gain_var.get():.1f} dB")
            
            # Detection settings are loaded when detection tab is created
            
        except Exception as e:
            print(f"Fehler beim Laden der GUI-Einstellungen: {e}")
    
    def setup_settings_auto_save(self):
        """Setzt automatisches Speichern bei Änderungen auf"""
        try:
            # Trace connection variables
            self.ip_var.trace_add('write', lambda *args: self.save_all_settings())
            self.user_var.trace_add('write', lambda *args: self.save_all_settings())
            self.password_var.trace_add('write', lambda *args: self.save_all_settings())
            
            # Trace streaming variables  
            self.stream_var.trace_add('write', lambda *args: self.save_all_settings())
            self.fps_var.trace_add('write', lambda *args: self.save_all_settings())
            self.quality_var.trace_add('write', lambda *args: self.save_all_settings())
            
            # Trace camera variables
            self.exposure_var.trace_add('write', lambda *args: self.save_all_settings())
            self.gain_var.trace_add('write', lambda *args: self.save_all_settings())
            self.flip_h_var.trace_add('write', lambda *args: self.save_all_settings())
            self.flip_v_var.trace_add('write', lambda *args: self.save_all_settings())
            
        except Exception as e:
            print(f"Fehler beim Setup des Auto-Save: {e}")
    
    def create_gui(self):
        """Erstellt die Benutzeroberfläche"""
        
        # Hauptframe
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Verbindungs-Frame
        connection_frame = ttk.LabelFrame(main_frame, text="Kamera-Verbindung", padding="5")
        connection_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # IP-Eingabe
        ttk.Label(connection_frame, text="IP-Adresse:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.ip_var = tk.StringVar()
        ip_entry = ttk.Entry(connection_frame, textvariable=self.ip_var, width=20)
        ip_entry.grid(row=0, column=1, padx=(0, 10))
        self.ip_var.trace_add('write', lambda *args: self.save_all_settings())
        
        # Benutzer-Eingabe
        ttk.Label(connection_frame, text="Benutzer:").grid(row=0, column=2, sticky=tk.W, padx=(10, 5))
        self.user_var = tk.StringVar()
        user_entry = ttk.Entry(connection_frame, textvariable=self.user_var, width=15)
        user_entry.grid(row=0, column=3, padx=(0, 10))
        self.user_var.trace_add('write', lambda *args: self.save_all_settings())
        
        # Passwort-Eingabe
        ttk.Label(connection_frame, text="Passwort:").grid(row=0, column=4, sticky=tk.W, padx=(10, 5))
        self.password_var = tk.StringVar()
        password_entry = ttk.Entry(connection_frame, textvariable=self.password_var, show="*", width=15)
        password_entry.grid(row=0, column=5, padx=(0, 10))
        self.password_var.trace_add('write', lambda *args: self.save_all_settings())
        
        # Verbinden-Button
        self.connect_btn = ttk.Button(connection_frame, text="Verbinden", command=self.connect_camera)
        self.connect_btn.grid(row=0, column=6, padx=(10, 0))
        
        # Status-Label
        self.status_var = tk.StringVar(value="Nicht verbunden")
        self.status_label = ttk.Label(connection_frame, textvariable=self.status_var, foreground="red")
        self.status_label.grid(row=1, column=0, columnspan=7, pady=(5, 0))
        
        # Tab-Control für verschiedene Ansichten
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Tab 1: Live-Streaming
        self.create_streaming_tab()
        
        # Tab 2: System-Monitor
        self.create_system_monitor_tab()
        
        # Tab 3: Objekterkennung
        if YOLO_AVAILABLE:
            self.create_detection_tab()

        # Tab 4: Gallery
        self.create_gallery_tab()
        
        # Grid-Gewichtung
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def create_streaming_tab(self):
        """Erstellt das Streaming-Tab"""
        streaming_tab = ttk.Frame(self.notebook)
        self.notebook.add(streaming_tab, text="Live-Streaming")
        
        # Streaming-Frame
        streaming_frame = ttk.LabelFrame(streaming_tab, text="Live-Streaming", padding="5")
        streaming_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Performance-Einstellungen Frame
        perf_frame = ttk.Frame(streaming_frame)
        perf_frame.grid(row=1, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # FPS-Einstellung
        ttk.Label(perf_frame, text="Live-FPS:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.fps_var = tk.IntVar()
        fps_scale = ttk.Scale(perf_frame, from_=5, to=30, variable=self.fps_var, 
                             orient="horizontal", length=100, command=self.update_target_fps)
        fps_scale.grid(row=0, column=1, padx=(0, 10))
        self.fps_label = ttk.Label(perf_frame, text="")
        self.fps_label.grid(row=0, column=2, padx=(0, 15))
        
        # Qualitäts-Einstellung
        ttk.Label(perf_frame, text="Qualität:").grid(row=0, column=3, sticky=tk.W, padx=(0, 5))
        self.quality_var = tk.IntVar()
        quality_scale = ttk.Scale(perf_frame, from_=30, to=95, variable=self.quality_var, 
                                 orient="horizontal", length=100, command=self.update_image_quality)
        quality_scale.grid(row=0, column=4, padx=(0, 10))
        self.quality_label = ttk.Label(perf_frame, text="")
        self.quality_label.grid(row=0, column=5, padx=(0, 10))
        
        # Performance-Anzeige
        self.perf_label = ttk.Label(perf_frame, text="FPS: -- | Latenz: --ms", foreground="blue")
        self.perf_label.grid(row=0, column=6, padx=(10, 0))
        
        # Stream-Auswahl
        ttk.Label(streaming_frame, text="Stream:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.stream_var = tk.StringVar()
        stream_combo = ttk.Combobox(streaming_frame, textvariable=self.stream_var, 
                                   values=["stream1", "stream2", "stream3", "live_images"], width=15, state="readonly")
        stream_combo.grid(row=0, column=1, padx=(0, 10))
        stream_combo.bind('<<ComboboxSelected>>', lambda e: self.save_all_settings())
        
        # Streaming-Buttons
        self.start_stream_btn = ttk.Button(streaming_frame, text="Stream starten", 
                                          command=self.start_streaming, state="disabled")
        self.start_stream_btn.grid(row=0, column=2, padx=(0, 5))
        
        self.stop_stream_btn = ttk.Button(streaming_frame, text="Stream stoppen", 
                                         command=self.stop_streaming, state="disabled")
        self.stop_stream_btn.grid(row=0, column=3, padx=(0, 5))
        
        # Einzelbild-Button
        self.capture_btn = ttk.Button(streaming_frame, text="Einzelbild aufnehmen",
                                     command=self.capture_image, state="disabled")
        self.capture_btn.grid(row=0, column=4, padx=(10, 0))

        # Workflow-Weiterleitung zur Labeling-App
        self.labeling_btn = ttk.Button(streaming_frame, text="Weiter zur Labeling App",
                                       command=self.open_labeling_app)
        self.labeling_btn.grid(row=0, column=5, padx=(10, 0))
        
        # Video-Display-Frame
        video_frame = ttk.LabelFrame(streaming_tab, text="Live-Bild", padding="5")
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Video-Label (für Bildanzeige)
        self.video_label = ttk.Label(video_frame, text="Kein Bild verfügbar",
                                    anchor="center", background="black", foreground="white")
        self.video_label.pack(expand=True, fill="both")

        # Kamera-Steuerung-Frame
        control_frame = ttk.LabelFrame(streaming_tab, text="Kamera-Steuerung", padding="5")
        control_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Belichtungszeit
        ttk.Label(control_frame, text="Belichtungszeit (µs):").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.exposure_var = tk.IntVar()
        exposure_scale = ttk.Scale(control_frame, from_=100, to=100000, variable=self.exposure_var, 
                                  orient="horizontal", length=200, command=self.update_exposure)
        exposure_scale.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.exposure_label = ttk.Label(control_frame, text="")
        self.exposure_label.grid(row=2, column=0, pady=(0, 10))
        
        # Gain
        ttk.Label(control_frame, text="Gain (dB):").grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        self.gain_var = tk.DoubleVar()
        gain_scale = ttk.Scale(control_frame, from_=0, to=30, variable=self.gain_var, 
                              orient="horizontal", length=200, command=self.update_gain)
        gain_scale.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.gain_label = ttk.Label(control_frame, text="")
        self.gain_label.grid(row=5, column=0, pady=(0, 10))
        
        # Flip-Optionen
        self.flip_h_var = tk.BooleanVar()
        flip_h_check = ttk.Checkbutton(control_frame, text="Horizontal spiegeln", 
                                      variable=self.flip_h_var, command=self.update_flip)
        flip_h_check.grid(row=6, column=0, sticky=tk.W, pady=(0, 5))
        self.flip_h_var.trace_add('write', lambda *args: self.save_all_settings())
        
        self.flip_v_var = tk.BooleanVar()
        flip_v_check = ttk.Checkbutton(control_frame, text="Vertikal spiegeln", 
                                      variable=self.flip_v_var, command=self.update_flip)
        flip_v_check.grid(row=7, column=0, sticky=tk.W, pady=(0, 10))
        self.flip_v_var.trace_add('write', lambda *args: self.save_all_settings())
        
        # Info-Textfeld
        info_frame = ttk.LabelFrame(control_frame, text="Geräteinformationen", padding="5")
        info_frame.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.info_text = tk.Text(info_frame, height=8, width=30, wrap=tk.WORD)
        info_scroll = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scroll.set)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Grid-Gewichtung
        streaming_tab.columnconfigure(0, weight=3)
        streaming_tab.columnconfigure(1, weight=1)
        streaming_tab.rowconfigure(1, weight=1)
        
        # Objekterkennungs-Status im Live-Streaming Tab
        detection_status_frame = ttk.LabelFrame(streaming_tab, text="Objekterkennung Status", padding="5")
        detection_status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.streaming_detection_status_label = ttk.Label(
            detection_status_frame,
            text="Objekterkennung: Inaktiv",
            foreground="orange",
        )
        self.streaming_detection_status_label.grid(row=0, column=0, padx=(0, 10))

        self.streaming_motion_status_label = ttk.Label(
            detection_status_frame,
            text="Motion: --",
            foreground="blue",
        )
        self.streaming_motion_status_label.grid(row=0, column=1, padx=(0, 10))

        self.streaming_objects_count_label = ttk.Label(
            detection_status_frame,
            text="Objekte: 0",
            foreground="green",
        )
        self.streaming_objects_count_label.grid(row=0, column=2)
    
    def create_system_monitor_tab(self):
        """Erstellt das System-Monitor-Tab"""
        monitor_tab = ttk.Frame(self.notebook)
        self.notebook.add(monitor_tab, text="Systemmonitor")
        
        # System-Monitor-Frame
        monitor_frame = ttk.LabelFrame(monitor_tab, text="Systemmonitor", padding="10")
        monitor_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Monitor-Buttons
        button_frame = ttk.Frame(monitor_frame)
        button_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_monitor_btn = ttk.Button(button_frame, text="Monitoring starten", 
                                           command=self.start_monitoring, state="disabled")
        self.start_monitor_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.stop_monitor_btn = ttk.Button(button_frame, text="Monitoring stoppen", 
                                          command=self.stop_monitoring, state="disabled")
        self.stop_monitor_btn.grid(row=0, column=1, padx=(0, 5))
        
        self.refresh_btn = ttk.Button(button_frame, text="Aktualisieren", 
                                     command=self.refresh_system_data, state="disabled")
        self.refresh_btn.grid(row=0, column=2, padx=(0, 5))
        
        # System-Gauges in 2x3 Grid
        # Gauges in einer Reihe
        self.temp_frame = self.create_gauge_frame(monitor_frame, "Temperature", "0.0 °C", "orange")
        self.temp_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.fps_frame = self.create_gauge_frame(monitor_frame, "FPS", "0.0", "teal")
        
        # Geräte-Informationen
        info_frame = ttk.LabelFrame(monitor_tab, text="Geräteinformationen", padding="10")
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        # Info-Labels
        info_labels = [
            ("Gerätename:", "device_name"),
            ("Gerätetyp:", "device_type"),
            ("Gerätemodell:", "device_model"),
            ("MAC Adresse:", "mac_address"),
            ("Seriennummer:", "serial_number"),
            ("Version:", "version"),
            ("Standort:", "location")
        ]
        
        self.info_labels = {}
        for i, (label_text, key) in enumerate(info_labels):
            row = i // 2
            col = (i % 2) * 2
            
            ttk.Label(info_frame, text=label_text).grid(row=row, column=col, sticky=tk.W, padx=(0, 5), pady=2)
            value_label = ttk.Label(info_frame, text="--", foreground="blue")
            value_label.grid(row=row, column=col+1, sticky=tk.W, padx=(0, 20), pady=2)
            self.info_labels[key] = value_label
        
        # Grid-Gewichtung
        monitor_tab.columnconfigure(0, weight=1)
        monitor_tab.rowconfigure(0, weight=1)
        monitor_frame.columnconfigure(0, weight=1)
        monitor_frame.columnconfigure(1, weight=1)
        monitor_frame.rowconfigure(1, weight=1)
        status_frame = ttk.Frame(monitor_tab)
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=10, pady=5)
        
        self.data_source_label = ttk.Label(status_frame, text="Datenquelle: --", foreground="gray")
        self.data_source_label.pack(side=tk.LEFT)
        
        # Live-Stream FPS Anzeige
        self.live_fps_info = ttk.Label(status_frame, text="Live-Stream: -- FPS", foreground="blue")
        self.live_fps_info.pack(side=tk.RIGHT)
    
    def create_detection_tab(self):
        """Erstellt das Objekterkennungs-Tab"""
        detection_tab = ttk.Frame(self.notebook)
        self.notebook.add(detection_tab, text="Objekterkennung")
        self.detection_tab = detection_tab
        
        # Hauptlayout
        main_frame = ttk.Frame(detection_tab)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Model und Dataset Auswahl
        setup_frame = ttk.LabelFrame(main_frame, text="Setup", padding="10")
        setup_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Model auswählen
        ttk.Label(setup_frame, text="YOLO Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.model_path_var = tk.StringVar()
        model_entry = ttk.Entry(setup_frame, textvariable=self.model_path_var, width=40, state="readonly")
        model_entry.grid(row=0, column=1, padx=(0, 5))
        model_btn = ttk.Button(setup_frame, text="Durchsuchen", command=self.browse_yolo_model)
        model_btn.grid(row=0, column=2)
        
        # YAML Dataset auswählen
        ttk.Label(setup_frame, text="Dataset YAML:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.yaml_path_var = tk.StringVar()
        yaml_entry = ttk.Entry(setup_frame, textvariable=self.yaml_path_var, width=40, state="readonly")
        yaml_entry.grid(row=1, column=1, padx=(0, 5), pady=(5, 0))
        yaml_btn = ttk.Button(setup_frame, text="Durchsuchen", command=self.browse_yaml_file)
        yaml_btn.grid(row=1, column=2, pady=(5, 0))
        
        # Detection Controls
        control_frame = ttk.LabelFrame(main_frame, text="Steuerung", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Detection Ein/Aus
        self.detection_var = tk.BooleanVar(value=self.detection_settings.get('enabled', False))
        detection_check = ttk.Checkbutton(control_frame, text="Objekterkennung aktivieren", 
                                         variable=self.detection_var, command=self.toggle_detection)
        detection_check.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Motion Threshold
        ttk.Label(control_frame, text="Motion Threshold:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        self.motion_var = tk.IntVar(value=self.detection_settings.get('motion_threshold', 110))
        motion_scale = ttk.Scale(control_frame, from_=50, to=200, variable=self.motion_var,
                                orient="horizontal", length=200, command=self.update_motion_threshold)
        motion_scale.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        self.motion_label = ttk.Label(control_frame, text=f"Wert: {self.motion_var.get()}")
        self.motion_label.grid(row=3, column=0, pady=(0, 10))
        self.motion_var.trace_add('write', lambda *args: self.save_detection_settings(False))
        
        # IoU Threshold
        ttk.Label(control_frame, text="IoU Threshold:").grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        self.iou_var = tk.DoubleVar(value=self.detection_settings.get('iou_threshold', 0.45))
        iou_scale = ttk.Scale(control_frame, from_=0.1, to=0.9, variable=self.iou_var,
                             orient="horizontal", length=200, command=self.update_iou_threshold)
        iou_scale.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        self.iou_label = ttk.Label(control_frame, text=f"Wert: {self.iou_var.get():.2f}")
        self.iou_label.grid(row=6, column=0, pady=(0, 10))
        self.iou_var.trace_add('write', lambda *args: self.save_detection_settings(False))
        
        # Einstellungen speichern/laden
        settings_btn_frame = ttk.Frame(control_frame)
        settings_btn_frame.grid(row=7, column=0, columnspan=2, pady=(10, 0))
        
        save_btn = ttk.Button(settings_btn_frame, text="Einstellungen speichern", 
                             command=self.save_detection_settings)
        save_btn.grid(row=0, column=0, padx=(0, 5))
        
        load_btn = ttk.Button(settings_btn_frame, text="Einstellungen laden", 
                             command=self.load_detection_settings_manual)
        load_btn.grid(row=0, column=1)
        
        # Class Thresholds Frame
        thresholds_frame = ttk.LabelFrame(main_frame, text="Klassen-Confidence", padding="10")
        thresholds_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollable frame für Klassen
        self.threshold_canvas = tk.Canvas(thresholds_frame, height=300)
        threshold_scrollbar = ttk.Scrollbar(thresholds_frame, orient="vertical", 
                                           command=self.threshold_canvas.yview)
        self.threshold_scrollable_frame = ttk.Frame(self.threshold_canvas)
        
        self.threshold_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.threshold_canvas.configure(scrollregion=self.threshold_canvas.bbox("all"))
        )
        
        self.threshold_canvas.create_window((0, 0), window=self.threshold_scrollable_frame, anchor="nw")
        self.threshold_canvas.configure(yscrollcommand=threshold_scrollbar.set)
        
        self.threshold_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        threshold_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Detection Status und Statistiken
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.detection_status_label = ttk.Label(status_frame, text="Status: Inaktiv", foreground="red")
        self.detection_status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.detection_stats_label = ttk.Label(status_frame, text="Erkennungen: --", foreground="blue")
        self.detection_stats_label.grid(row=0, column=1, sticky=tk.E)
        
        self.motion_status_label = ttk.Label(status_frame, text="Motion: --", foreground="gray")
        self.motion_status_label.grid(row=1, column=0, sticky=tk.W)
        
        # Grid-Gewichtung
        detection_tab.columnconfigure(0, weight=1)
        detection_tab.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        control_frame.columnconfigure(0, weight=1)
        thresholds_frame.columnconfigure(0, weight=1)
        thresholds_frame.rowconfigure(0, weight=1)
        
        # Initiale Einstellungen laden
        self.load_saved_paths()
    
    def create_gauge_frame(self, parent, title, initial_value, color):
        """Erstellt ein Gauge-Frame für Systemwerte"""
        frame = ttk.LabelFrame(parent, text=title, padding="10")
        
        # Kreis-Simulation mit Canvas
        canvas = tk.Canvas(frame, width=120, height=120, bg='white', highlightthickness=0)
        canvas.grid(row=0, column=0, pady=(0, 10))
        
        # Kreis-Hintergrund
        canvas.create_oval(10, 10, 110, 110, outline='lightgray', width=8)
        
        # Wert-Label
        value_label = ttk.Label(frame, text=initial_value, font=('Arial', 12, 'bold'))
        value_label.grid(row=1, column=0)
        
        # Elemente für späteren Zugriff speichern
        frame.canvas = canvas
        frame.value_label = value_label
        frame.color = color
        frame.arc_id = None
        
        return frame
    
    def update_gauge(self, gauge_frame, value, max_value=100, unit=""):
        """Aktualisiert ein Gauge mit neuem Wert"""
        try:
            canvas = gauge_frame.canvas
            value_label = gauge_frame.value_label
            color = gauge_frame.color
            
            # Alten Bogen löschen
            if gauge_frame.arc_id:
                canvas.delete(gauge_frame.arc_id)
            
            # Neuen Bogen zeichnen
            if value > 0:
                extent = -(value / max_value) * 360  # Negativ für Uhrzeigersinn
                gauge_frame.arc_id = canvas.create_arc(10, 10, 110, 110, start=90, extent=extent, 
                                                      outline=color, width=8, style='arc')
            
            # Wert aktualisieren
            if isinstance(value, float):
                value_label.configure(text=f"{value:.1f} {unit}")
            else:
                value_label.configure(text=f"{value} {unit}")
                
        except Exception as e:
            print(f"Fehler beim Aktualisieren des Gauges: {e}")
    
    def auto_connect(self):
        """Automatische Verbindung wenn Daten vorhanden"""
        self.connect_camera()
    
    def connect_camera(self):
        """Verbindung zur Kamera herstellen"""
        ip = self.ip_var.get().strip()
        user = self.user_var.get().strip()
        password = self.password_var.get().strip()
        
        if not ip:
            messagebox.showerror("Fehler", "Bitte IP-Adresse eingeben!")
            return
        
        try:
            self.status_var.set("Verbinde...")
            self.status_label.configure(foreground="orange")
            self.root.update()
            
            # REST-Verbindung erstellen
            self.rest_connection = NXTRestConnection(ip, user, password)
            
            # Handler erstellen
            self.camera_handler = NXTCameraHandlerBase(self.rest_connection)
            self.streaming_handler = NXTStreamingHandler(self.rest_connection)
            
            # Verbindung testen
            device_info = self.camera_handler.get_device_info()
            
            # Konfiguration speichern
            self.save_connection_config(ip, user, password)
            
            # GUI aktualisieren
            self.status_var.set("Verbunden")
            self.status_label.configure(foreground="green")
            self.connect_btn.configure(text="Trennen", command=self.disconnect_camera)
            self.start_stream_btn.configure(state="normal")
            self.capture_btn.configure(state="normal")
            
            # System-Monitor-Buttons aktivieren
            self.start_monitor_btn.configure(state="normal")
            self.refresh_btn.configure(state="normal")
            
            # Geräteinformationen anzeigen
            self.show_device_info(device_info)
            
            # Aktuelle Kamera-Einstellungen laden
            self.load_camera_settings()
            
            # Streaming-Verfügbarkeit prüfen
            self.check_streaming_availability()
            
            # Initiale Systemdaten laden
            self.refresh_system_data()
            
        except Exception as e:
            self.status_var.set("Verbindung fehlgeschlagen")
            self.status_label.configure(foreground="red")
            messagebox.showerror("Verbindungsfehler", f"Fehler beim Verbinden: {str(e)}")
    
    def disconnect_camera(self):
        """Verbindung zur Kamera trennen"""
        self.stop_streaming()
        self.stop_monitoring()
        
        self.rest_connection = None
        self.camera_handler = None
        self.streaming_handler = None
        
        self.status_var.set("Nicht verbunden")
        self.status_label.configure(foreground="red")
        self.connect_btn.configure(text="Verbinden", command=self.connect_camera)
        self.start_stream_btn.configure(state="disabled")
        self.stop_stream_btn.configure(state="disabled")
        self.capture_btn.configure(state="disabled")
        
        # System-Monitor-Buttons deaktivieren
        self.start_monitor_btn.configure(state="disabled")
        self.stop_monitor_btn.configure(state="disabled")
        self.refresh_btn.configure(state="disabled")
        
        self.info_text.delete(1.0, tk.END)
        self.video_label.configure(image="", text="Kein Bild verfügbar")
    
    def save_connection_config(self, ip, user, password):
        """Verbindungsdaten in Einstellungen speichern"""
        try:
            self.settings["connection"]["ip"] = ip
            self.settings["connection"]["user"] = user  
            self.settings["connection"]["password"] = password
            self.save_all_settings()
        except Exception as e:
            print(f"Fehler beim Speichern der Konfiguration: {e}")
    
    def show_device_info(self, device_info):
        """Geräteinformationen anzeigen"""
        try:
            info_data = device_info.to_json()
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, "Geräteinformationen:\n\n")
            for key, value in info_data.items():
                self.info_text.insert(tk.END, f"{key}: {value}\n")
        except Exception as e:
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, f"Fehler beim Laden der Geräteinformationen: {e}")
    
    def load_camera_settings(self):
        """Aktuelle Kamera-Einstellungen laden"""
        try:
            # Hier könnten Sie die aktuellen Kamera-Einstellungen laden
            # und die GUI-Elemente entsprechend aktualisieren
            pass
        except Exception as e:
            print(f"Fehler beim Laden der Kamera-Einstellungen: {e}")
    
    def check_streaming_availability(self):
        """Prüft, ob Streaming verfügbar ist"""
        try:
            # Versuche, verfügbare Streams abzurufen
            available_streams = self.streaming_handler.get_available_streams()
            self.streaming_available = True
            print("Streaming verfügbar:", available_streams)
        except Exception as e:
            self.streaming_available = False
            print(f"Streaming nicht verfügbar: {e}")
            # Stream-Auswahl auf "Einzelbilder" ändern
            self.stream_var.set("live_images")
            # Combobox-Werte aktualisieren
            stream_combo = None
            for widget in self.root.winfo_children():
                if hasattr(widget, 'winfo_children'):
                    for child in widget.winfo_children():
                        if hasattr(child, 'winfo_children'):
                            for grandchild in child.winfo_children():
                                if isinstance(grandchild, ttk.Combobox):
                                    grandchild.configure(values=["live_images"])
                                    break
    
    def start_streaming(self):
        """Live-Streaming starten"""
        if not self.streaming_handler:
            messagebox.showerror("Fehler", "Keine Verbindung zur Kamera!")
            return
        
        try:
            stream_name = self.stream_var.get()
            
            if self.streaming_available and stream_name in ["stream1", "stream2", "stream3"]:
                # Echtes Streaming verwenden
                if stream_name == "stream1":
                    stream_info = self.streaming_handler.get_stream1()
                elif stream_name == "stream2":
                    stream_info = self.streaming_handler.get_stream2()
                else:
                    stream_info = self.streaming_handler.get_stream3()
                
                self.current_stream_url = stream_info.URL
                
                if not self.current_stream_url:
                    messagebox.showerror("Fehler", "Stream-URL nicht verfügbar!")
                    return
            else:
                # Fallback: Live-Bilder über kontinuierliche Einzelbildaufnahme
                self.current_stream_url = "live_images"
            
            # Streaming starten
            self.streaming_active = True
            self.start_stream_btn.configure(state="disabled")
            self.stop_stream_btn.configure(state="normal")
            
            # Entsprechenden Stream-Thread starten
            if self.streaming_available and stream_name in ["stream1", "stream2", "stream3"]:
                self.stream_thread = threading.Thread(target=self.stream_worker, daemon=True)
            else:
                self.stream_thread = threading.Thread(target=self.live_image_worker, daemon=True)
            self.stream_thread.start()
            
        except Exception as e:
            messagebox.showerror("Streaming-Fehler", f"Fehler beim Starten des Streams: {str(e)}")
    
    def stop_streaming(self):
        """Live-Streaming stoppen"""
        self.streaming_active = False
        
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2)
        
        # Performance-Counter zurücksetzen
        self.fps_counter = 0
        self.fps_last_time = time.time()
        self.perf_label.configure(text="FPS: -- | Latenz: --ms")
        
        self.start_stream_btn.configure(state="normal")
        self.stop_stream_btn.configure(state="disabled")
        self.video_label.configure(image="", text="Stream gestoppt")
    
    def stream_worker(self):
        """Worker-Thread für Live-Streaming"""
        while self.streaming_active:
            try:
                if self.current_stream_url:
                    # MJPEG-Stream lesen
                    stream_url = f"http://{self.rest_connection.ip}{self.current_stream_url}"
                    
                    # Einzelbild vom Stream abrufen
                    response = urllib.request.urlopen(stream_url, timeout=5)
                    image_data = response.read()
                    
                    # Bild verarbeiten und anzeigen
                    image = Image.open(io.BytesIO(image_data))
                    image = self.resize_image_to_label(image)
                    photo = ImageTk.PhotoImage(image)
                    
                    # GUI-Update im Hauptthread
                    self.root.after(0, self.update_video_display, photo)
                
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                if self.streaming_active:
                    print(f"Stream-Fehler: {e}")
                    self.root.after(0, self.handle_stream_error, str(e))
                break
    
    def live_image_worker(self):
        """Optimierter Worker-Thread für Live-Bilder mit Performance-Verbesserungen"""
        
        # Performance-Optimierungen
        from concurrent.futures import ThreadPoolExecutor
        import queue
        
        # Bildpuffer für smoothere Darstellung
        image_queue = queue.Queue(maxsize=3)
        
        def capture_image():
            """Einzelbild-Aufnahme in separatem Thread"""
            try:
                start_time = time.time()
                
                # Dynamische Qualitätsanpassung basierend auf Performance
                current_quality = self.adaptive_quality
                
                # Optimierte HTTP-Parameter
                image_header = self.rest_connection.get_image_accept_header_by_filename("temp.jpg")
                header = {
                    'Accept': image_header.value,
                    'Connection': 'keep-alive',  # HTTP Keep-Alive für bessere Performance
                    'Cache-Control': 'no-cache'
                }
                params = {'quality': current_quality}
                
                # Bild von Kamera abrufen
                result = self.rest_connection.get('/camera/image', params=params, additional_headers=header)
                
                capture_time = time.time() - start_time
                
                # Adaptive Qualitätsregelung
                if capture_time > 0.1:  # Wenn zu langsam, Qualität reduzieren
                    self.adaptive_quality = max(30, self.adaptive_quality - 5)
                elif capture_time < 0.05:  # Wenn schnell genug, Qualität erhöhen
                    self.adaptive_quality = min(self.quality_var.get(), self.adaptive_quality + 2)
                
                return result.get_response_data(), capture_time * 1000  # Latenz in ms
                
            except Exception as e:
                print(f"Bildaufnahme-Fehler: {e}")
                return None, 0
        
        def process_and_display():
            """Bildverarbeitung und Display in separatem Thread"""
            try:
                if not image_queue.empty():
                    image_data, latency = image_queue.get_nowait()
                    if image_data:
                        # Optimierte Bildverarbeitung
                        image = Image.open(io.BytesIO(image_data))
                        
                        # Objekterkennung anwenden falls aktiviert
                        if self.detection_enabled and YOLO_AVAILABLE:
                            # PIL zu OpenCV konvertieren
                            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                            # Detection durchführen
                            cv_image = self.process_detection_on_frame(cv_image)
                            # Zurück zu PIL konvertieren
                            image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
                        
                        # Intelligente Skalierung abhängig von Label-Größe
                        image = self.resize_image_to_label(image)
                        
                        photo = ImageTk.PhotoImage(image)
                        
                        # Performance-Counter aktualisieren
                        self.update_performance_stats(latency)
                        
                        # GUI-Update im Hauptthread
                        self.root.after(0, self.update_video_display, photo)
                        
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Bildverarbeitung-Fehler: {e}")
        
        # Multi-Threading für parallele Verarbeitung
        with ThreadPoolExecutor(max_workers=2) as executor:
            while self.streaming_active:
                try:
                    frame_start_time = time.time()
                    
                    # Parallele Bildaufnahme starten
                    future_capture = executor.submit(capture_image)
                    
                    # Gleichzeitig vorheriges Bild verarbeiten
                    process_and_display()
                    
                    # Auf Bildaufnahme warten
                    image_data, latency = future_capture.result(timeout=1.0)
                    
                    if image_data:
                        # Bild in Queue einreihen (non-blocking)
                        try:
                            image_queue.put_nowait((image_data, latency))
                        except queue.Full:
                            # Ältestes Bild verwerfen wenn Buffer voll
                            try:
                                image_queue.get_nowait()
                                image_queue.put_nowait((image_data, latency))
                            except queue.Empty:
                                pass
                    
                    # Adaptive Frame-Rate Kontrolle
                    target_frame_time = 1.0 / self.target_fps
                    frame_time = time.time() - frame_start_time
                    
                    if frame_time < target_frame_time:
                        time.sleep(target_frame_time - frame_time)
                    
                except Exception as e:
                    if self.streaming_active:
                        print(f"Live-Stream-Fehler: {e}")
                        self.root.after(0, self.handle_stream_error, str(e))
                    break
        
        while self.streaming_active:
            try:
                # Verarbeitungsschleife für Queue-basierte Anzeige
                process_and_display()
                time.sleep(0.01)  # Kurze Pause für GUI-Responsivität
                
            except Exception as e:
                if self.streaming_active:
                    print(f"Live-Image-Fehler: {e}")
                    self.root.after(0, self.handle_stream_error, str(e))
                break
    
    def update_video_display(self, photo):
        """Video-Display aktualisieren"""
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo  # Referenz behalten

    def resize_image_to_label(self, image):
        """Skaliert ein PIL-Image auf die Größe des Video-Labels und bewahrt das Seitenverhältnis"""
        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()
        if label_w <= 1 or label_h <= 1:
            label_w, label_h = 640, 480

        img_w, img_h = image.size
        img_ratio = img_w / img_h
        label_ratio = label_w / label_h

        if label_ratio > img_ratio:
            new_h = label_h
            new_w = int(new_h * img_ratio)
        else:
            new_w = label_w
            new_h = int(new_w / img_ratio)

        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    def handle_stream_error(self, error_msg):
        """Stream-Fehler behandeln"""
        self.stop_streaming()
        if not self.streaming_available:
            self.video_label.configure(text="Live-Bilder werden über Einzelbildaufnahme dargestellt")
        else:
            self.video_label.configure(text=f"Stream-Fehler: {error_msg}")
    
    def update_performance_stats(self, latency):
        """Performance-Statistiken aktualisieren"""
        current_time = time.time()
        self.fps_counter += 1
        
        # FPS alle Sekunde aktualisieren
        if current_time - self.fps_last_time >= 1.0:
            fps = self.fps_counter / (current_time - self.fps_last_time)
            self.current_live_fps = round(fps, 1)  # Aktuelle Live-Stream FPS speichern
            self.fps_counter = 0
            self.fps_last_time = current_time
            
            # Performance-Anzeige aktualisieren
            self.perf_label.configure(text=f"FPS: {fps:.1f} | Latenz: {latency:.0f}ms | Q: {self.adaptive_quality}")
            print(f"📈 Live-Stream Performance: FPS={self.current_live_fps}, Latenz={latency:.0f}ms")
            
            # Live-FPS Info im Systemmonitor aktualisieren
            if hasattr(self, 'live_fps_info'):
                self.live_fps_info.configure(text=f"Live-Stream: {self.current_live_fps} FPS")
    
    def update_target_fps(self, value):
        """Ziel-FPS aktualisieren"""
        self.target_fps = int(float(value))
        self.fps_label.configure(text=f"{self.target_fps} FPS")
    
    def update_image_quality(self, value):
        """Bildqualität aktualisieren"""
        quality = int(float(value))
        self.quality_label.configure(text=f"{quality}%")
        self.adaptive_quality = quality
    
    def capture_image(self):
        """Einzelbild aufnehmen"""
        if not self.camera_handler:
            messagebox.showerror("Fehler", "Keine Verbindung zur Kamera!")
            return
        
        try:
            raw_dir = self.project_manager.get_raw_images_dir()
            filename = raw_dir / f"capture_{int(time.time())}.jpg"
            self.camera_handler.save_camera_image_latest(str(filename))
            messagebox.showinfo("Erfolg", f"Bild gespeichert als: {filename.name}")
            self.refresh_gallery()
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Aufnehmen: {str(e)}")
    
    def update_exposure(self, value):
        """Belichtungszeit aktualisieren"""
        exposure = int(float(value))
        self.exposure_label.configure(text=f"{exposure} µs")
        
        if self.camera_handler:
            try:
                self.camera_handler.set_camera_setting("ExposureTime", exposure)
            except Exception as e:
                print(f"Fehler beim Setzen der Belichtungszeit: {e}")
    
    def update_gain(self, value):
        """Gain aktualisieren"""
        gain = float(value)
        self.gain_label.configure(text=f"{gain:.1f} dB")
        
        if self.camera_handler:
            try:
                self.camera_handler.set_camera_setting("Gain", gain)
            except Exception as e:
                print(f"Fehler beim Setzen des Gains: {e}")
    
    def update_flip(self):
        """Spiegelung aktualisieren"""
        if self.camera_handler:
            try:
                self.camera_handler.set_camera_setting("FlipHorizontal", self.flip_h_var.get())
                self.camera_handler.set_camera_setting("FlipVertical", self.flip_v_var.get())
            except Exception as e:
                print(f"Fehler beim Setzen der Spiegelung: {e}")

    # ==================== Gallery ====================
    def create_gallery_tab(self):
        """Erstellt das Gallery-Tab zum Durchsehen der aufgenommenen Bilder"""
        gallery_tab = ttk.Frame(self.notebook)
        self.notebook.add(gallery_tab, text="Gallery")

        list_frame = ttk.Frame(gallery_tab)
        list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.gallery_listbox = tk.Listbox(list_frame, width=30)
        self.gallery_listbox.pack(side=tk.LEFT, fill=tk.Y)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.gallery_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.gallery_listbox.configure(yscrollcommand=scrollbar.set)
        self.gallery_listbox.bind("<<ListboxSelect>>", self.display_selected_image)

        image_frame = ttk.Frame(gallery_tab)
        image_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.gallery_image_label = ttk.Label(image_frame)
        self.gallery_image_label.pack(expand=True, fill=tk.BOTH)

        btn_frame = ttk.Frame(gallery_tab)
        btn_frame.pack(side=tk.BOTTOM, pady=5)
        ttk.Button(btn_frame, text="Bild löschen", command=self.delete_selected_image).pack()

        self.gallery_photo = None
        self.refresh_gallery()

    def refresh_gallery(self):
        """Lädt Dateiliste aus dem Raw-Images-Verzeichnis"""
        raw_dir = self.project_manager.get_raw_images_dir()
        self.gallery_listbox.delete(0, tk.END)
        for img_path in sorted(raw_dir.glob('*')):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                self.gallery_listbox.insert(tk.END, img_path.name)

    def display_selected_image(self, event=None):
        """Zeigt ausgewähltes Bild in der Vorschau"""
        selection = self.gallery_listbox.curselection()
        if not selection:
            return
        filename = self.gallery_listbox.get(selection[0])
        img_path = self.project_manager.get_raw_images_dir() / filename
        try:
            img = Image.open(img_path)
            img.thumbnail((600, 600))
            self.gallery_photo = ImageTk.PhotoImage(img)
            self.gallery_image_label.configure(image=self.gallery_photo)
        except Exception as e:
            self.gallery_image_label.configure(text=f"Fehler beim Laden: {e}")

    def delete_selected_image(self):
        """Löscht ausgewähltes Bild aus dem Raw-Ordner"""
        selection = self.gallery_listbox.curselection()
        if not selection:
            return
        filename = self.gallery_listbox.get(selection[0])
        img_path = self.project_manager.get_raw_images_dir() / filename
        if messagebox.askyesno("Löschen", f"Bild '{filename}' wirklich löschen?"):
            try:
                img_path.unlink()
                self.refresh_gallery()
                self.gallery_image_label.configure(image="")
            except Exception as e:
                messagebox.showerror("Fehler", f"Bild konnte nicht gelöscht werden: {e}")

    # ==================== Workflow Navigation ====================
    def open_labeling_app(self):
        """Speichert Einstellungen, schließt die Kamera-App und öffnet die Labeling-App"""
        self.save_all_settings()
        if self.streaming_active:
            self.stop_streaming()
        if self.monitor_active:
            self.stop_monitoring()
        try:
            subprocess.Popen([
                sys.executable,
                "-m",
                "gui.image_labeling",
                str(self.settings_dir),
            ])
        except Exception as e:
            messagebox.showerror("Fehler", f"Labeling-App konnte nicht gestartet werden: {e}")
        self.root.destroy()
    
    def start_monitoring(self):
        """System-Monitoring starten"""
        if not self.camera_handler:
            messagebox.showerror("Fehler", "Keine Verbindung zur Kamera!")
            return
        
        # Einmalige Endpunkt-Analyse beim ersten Start
        if not hasattr(self, '_endpoints_analyzed'):
            self.test_available_endpoints()
            self._endpoints_analyzed = True
        
        self.monitor_active = True
        self.start_monitor_btn.configure(state="disabled")
        self.stop_monitor_btn.configure(state="normal")
        
        # Monitor-Thread starten
        self.monitor_thread = threading.Thread(target=self.monitor_worker, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """System-Monitoring stoppen"""
        self.monitor_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        
        self.start_monitor_btn.configure(state="normal")
        self.stop_monitor_btn.configure(state="disabled")
    
    def monitor_worker(self):
        """Worker-Thread für System-Monitoring"""
        while self.monitor_active:
            try:
                self.refresh_system_data()
                time.sleep(2)  # Alle 2 Sekunden aktualisieren
            except Exception as e:
                print(f"Monitor-Worker-Fehler: {e}")
                time.sleep(5)
    
    def refresh_system_data(self):
        """Systemdaten von der Kamera abrufen und anzeigen"""
        if not self.camera_handler:
            return
        
        try:
            # Verschiedene System-Endpunkte versuchen
            system_data = {}
            
            print("🔍 Debug: Systemdaten werden abgerufen...")
            
            # Grundlegende Geräteinformationen
            try:
                device_info = self.camera_handler.get_device_info().to_json()
                system_data.update(device_info)
                print(f"✅ DeviceInfo erfolgreich: {list(device_info.keys())}")
            except Exception as e:
                print(f"❌ DeviceInfo Fehler: {e}")
            
            # Verschiedene mögliche System-Endpunkte testen
            system_endpoints = [
                '/system',
                '/system/status', 
                '/system/monitor',
                '/system/info',
                '/devicestatus',
                '/status',
                '/monitor',
                '/statistics',
                '/health',
                '/performance'
            ]
            
            successful_endpoints = []
            
            for endpoint in system_endpoints:
                try:
                    result = self.rest_connection.get(endpoint)
                    endpoint_data = result.to_json()
                    if endpoint_data:
                        system_data.update(endpoint_data)
                        successful_endpoints.append(endpoint)
                        print(f"✅ {endpoint} erfolgreich: {list(endpoint_data.keys())}")
                    else:
                        print(f"⚠️ {endpoint} leer")
                except Exception as e:
                    print(f"❌ {endpoint} Fehler: {e}")
            
            print(f"📊 Erfolgreiche Endpunkte: {successful_endpoints}")
            print(f"🗄️ Alle verfügbaren Daten: {list(system_data.keys())}")
            
            # Falls keine echten Systemdaten gefunden wurden, versuche Kamera-spezifische Endpunkte
            if not any(key in system_data for key in ['CPU', 'CpuUsage', 'MemoryUsage', 'Temperature']):
                print("⚠️ Keine Standard-Systemdaten gefunden, versuche Kamera-spezifische Endpunkte...")
                
                # Kamera-spezifische Endpunkte testen
                camera_endpoints = [
                    '/camera/status',
                    '/camera/info',
                    '/camera/temperature',
                    '/network',
                    '/network/status'
                ]
                
                for endpoint in camera_endpoints:
                    try:
                        result = self.rest_connection.get(endpoint)
                        endpoint_data = result.to_json()
                        if endpoint_data:
                            system_data.update(endpoint_data)
                            print(f"✅ {endpoint} erfolgreich: {list(endpoint_data.keys())}")
                    except Exception as e:
                        print(f"❌ {endpoint} Fehler: {e}")
            
            # Verfügbare Endpunkte über OPTIONS ermitteln
            try:
                available_endpoints = self.discover_available_endpoints()
                print(f"🔍 Verfügbare Root-Endpunkte: {available_endpoints}")
            except Exception as e:
                print(f"❌ Endpunkt-Discovery Fehler: {e}")
            
            else:
                # Echte Live-Stream FPS zu echten Daten hinzufügen
                system_data['LiveStreamFPS'] = self.current_live_fps
                system_data['MockDataActive'] = False
            
            # GUI im Hauptthread aktualisieren
            self.root.after(0, self.update_system_display, system_data)
            
        except Exception as e:
            print(f"❌ Kritischer Fehler beim Abrufen der Systemdaten: {e}")
    
    def discover_available_endpoints(self):
        """Ermittelt verfügbare API-Endpunkte"""
        try:
            # Root-Endpunkte testen
            root_options = self.rest_connection.options('/').to_json()
            if 'Objects' in root_options:
                return root_options['Objects']
            elif 'objects' in root_options:
                return root_options['objects']
            else:
                return list(root_options.keys()) if root_options else []
        except Exception as e:
            print(f"❌ Root-Options Fehler: {e}")
            return []
    
    def update_system_display(self, data):
        """Systemdaten in der GUI anzeigen"""
        try:
            # CPU
            cpu_value = None
            for key in ['CPU', 'CpuUsage', 'cpu', 'cpu_usage']:
                if key in data and data[key] is not None:
                    cpu_value = data[key]
                    break

            if cpu_value is not None:
                self.update_gauge(self.cpu_frame, cpu_value, 100, "%")
                print(f"📊 CPU aktualisiert: {cpu_value}%")
            else:
                print("❌ CPU: Keine Daten verfügbar")

            # RAM
            ram_value = None
            for key in ['RAM', 'MemoryUsage', 'memory', 'memory_usage']:
                if key in data and data[key] is not None:
                    ram_value = data[key]
                    break

            if ram_value is not None:
                self.update_gauge(self.ram_frame, ram_value, 100, "%")
                print(f"📊 RAM aktualisiert: {ram_value}%")
            else:
                print("❌ RAM: Keine Daten verfügbar")

            # Disk
            disk_value = None
            for key in ['Disk', 'DiskUsage', 'disk', 'storage_usage']:
                if key in data and data[key] is not None:
                    disk_value = data[key]
                    break

            if disk_value is not None:
                self.update_gauge(self.disk_frame, disk_value, 100, "%")
                print(f"📊 Disk aktualisiert: {disk_value}%")
            else:
                print("❌ Disk: Keine Daten verfügbar")

            # Temperature
            temp_value = None
            for key in ['Temperature', 'CpuTemperature', 'temperature', 'cpu_temp']:
                if key in data and data[key] is not None:
                    temp_value = data[key]
                    break

            if temp_value is not None:
                self.update_gauge(self.temp_frame, temp_value, 100, "°C")
                print(f"📊 Temperature aktualisiert: {temp_value}°C")
            else:
                print("❌ Temperature: Keine Daten verfügbar")

            # Voltage
            voltage_value = None
            for key in ['Voltage', 'SupplyVoltage', 'voltage', 'supply_voltage']:
                if key in data and data[key] is not None:
                    voltage_value = data[key]
                    break

            if voltage_value is not None:
                self.update_gauge(self.voltage_frame, voltage_value, 15, "V")
                print(f"📊 Voltage aktualisiert: {voltage_value}V")
            else:
                print("❌ Voltage: Keine Daten verfügbar")

            # FPS (Live-Stream FPS, nicht Kamera-interne FPS)
            fps_value = data.get('LiveStreamFPS', None)

            if fps_value is not None:
                self.update_gauge(self.fps_frame, fps_value, 30, "")
                print(f"📊 Live-Stream FPS aktualisiert: {fps_value}")
            else:
                print("❌ FPS: Keine Daten verfügbar")

            # Status-Label aktualisieren
            if hasattr(self, 'data_source_label'):
                if data.get('MockDataActive', False):
                    self.data_source_label.configure(text="Datenquelle: Mock-Daten (Demo)", foreground="orange")
                else:
                    self.data_source_label.configure(text="Datenquelle: Live API-Daten", foreground="green")

            # Geräteinformationen aktualisieren
            info_mapping = {
                'device_name': ['DeviceName', 'Name', 'Hostname', 'hostname', 'device_name'],
                'device_type': ['DeviceType', 'Type', 'Model', 'model', 'device_type'],
                'device_model': ['DeviceModel', 'Model', 'ProductName', 'product_name', 'device_model'],
                'mac_address': ['MacAddress', 'MAC', 'NetworkMAC', 'mac_address', 'mac'],
                'serial_number': ['SerialNumber', 'Serial', 'DeviceSerial', 'serial_number', 'serial'],
                'version': ['Version', 'FirmwareVersion', 'SoftwareVersion', 'version', 'firmware_version'],
                'location': ['Location', 'Standort', 'Site', 'location', 'site']
            }

            for key, possible_keys in info_mapping.items():
                value = "--"
                for possible_key in possible_keys:
                    if possible_key in data:
                        value = str(data[possible_key])
                        print(f"📋 {key} gefunden als {possible_key}: {value}")
                        break

                if key in self.info_labels:
                    self.info_labels[key].configure(text=value)

        except Exception as e:
            print(f"❌ Fehler beim Aktualisieren der Systemanzeige: {e}")
    
    def test_available_endpoints(self):
        """Testet alle verfügbaren API-Endpunkte (Debug-Funktion)"""
        if not self.rest_connection:
            return
        
        print("🔍 ==> ENDPUNKT-ANALYSE GESTARTET <==")
        
        # Bekannte Basis-Endpunkte testen
        base_endpoints = [
            '/', '/camera', '/deviceinfo', '/system', '/network', 
            '/streaming', '/status', '/monitor', '/health', '/statistics'
        ]
        
        for endpoint in base_endpoints:
            try:
                # OPTIONS-Request für Verfügbarkeit
                options_result = self.rest_connection.options(endpoint)
                print(f"✅ {endpoint} - OPTIONS verfügbar")
                
                # GET-Request für Daten
                get_result = self.rest_connection.get(endpoint)
                data = get_result.to_json()
                if data:
                    print(f"📊 {endpoint} - Daten: {list(data.keys())}")
                    if isinstance(data, dict) and len(str(data)) < 500:
                        print(f"    Content: {data}")
                else:
                    print(f"⚠️ {endpoint} - Keine Daten")
                    
            except Exception as e:
                print(f"❌ {endpoint} - Fehler: {type(e).__name__}")
        
        print("🔍 ==> ENDPUNKT-ANALYSE BEENDET <==")
    
    # YOLO Detection Methods
    def browse_yolo_model(self):
        """YOLO Model-Datei auswählen"""
        from tkinter import filedialog
        
        file_path = filedialog.askopenfilename(
            title="YOLO Model auswählen",
            filetypes=[("PyTorch Models", "*.pt"), ("Alle Dateien", "*.*")]
        )
        
        if file_path:
            self.model_path_var.set(file_path)
            self.check_detection_ready()
    
    def browse_yaml_file(self):
        """YAML Dataset-Datei auswählen"""
        from tkinter import filedialog
        
        file_path = filedialog.askopenfilename(
            title="Dataset YAML auswählen",
            filetypes=[("YAML Files", "*.yaml *.yml"), ("Alle Dateien", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if 'names' not in data:
                    messagebox.showerror("Fehler", "YAML-Datei muss 'names' Feld enthalten!")
                    return
                
                self.class_names = data['names']
                if isinstance(self.class_names, list):
                    self.class_names = {i: name for i, name in enumerate(self.class_names)}
                
                self.yaml_path_var.set(file_path)
                self.create_class_threshold_widgets()
                self.check_detection_ready()
                
                print(f"✅ YAML geladen: {len(self.class_names)} Klassen")
                
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler beim Laden der YAML-Datei: {e}")
    
    def create_class_threshold_widgets(self):
        """Erstellt Widgets für Klassen-Confidence-Thresholds"""
        # Alte Widgets entfernen
        for widget in self.threshold_scrollable_frame.winfo_children():
            widget.destroy()
        
        self.class_threshold_vars = {}
        self.class_threshold_labels = {}
        
        for i, (class_id, class_name) in enumerate(self.class_names.items()):
            # Frame für jede Klasse
            class_frame = ttk.Frame(self.threshold_scrollable_frame)
            class_frame.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=2)
            
            # Label
            label = ttk.Label(class_frame, text=f"Klasse {class_id} ({class_name}):")
            label.grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
            
            # Threshold Variable
            saved_threshold = self.detection_settings.get('class_thresholds', {}).get(str(class_id), 0.7)
            threshold_var = tk.DoubleVar(value=saved_threshold)
            self.class_threshold_vars[class_id] = threshold_var
            threshold_var.trace_add('write', lambda *args: self.save_detection_settings(False))

            # Scale
            scale = ttk.Scale(class_frame, from_=0.1, to=0.95, variable=threshold_var,
                             orient="horizontal", length=150, 
                             command=lambda v, cid=class_id: self.update_class_threshold(cid, v))
            scale.grid(row=0, column=1, padx=(0, 5))
            
            # Value Label
            value_label = ttk.Label(class_frame, text=f"{saved_threshold:.2f}")
            value_label.grid(row=0, column=2)
            self.class_threshold_labels[class_id] = value_label
            
            class_frame.columnconfigure(1, weight=1)
        
        self.threshold_scrollable_frame.columnconfigure(0, weight=1)
    
    def update_class_threshold(self, class_id, value):
        """Aktualisiert Klassen-Threshold"""
        threshold = float(value)
        if class_id in self.class_threshold_labels:
            self.class_threshold_labels[class_id].configure(text=f"{threshold:.2f}")
    
    def update_motion_threshold(self, value):
        """Motion Threshold aktualisieren"""
        self.motion_threshold = int(float(value))
        self.motion_label.configure(text=f"Wert: {self.motion_threshold}")
    
    def update_iou_threshold(self, value):
        """IoU Threshold aktualisieren"""
        self.iou_label.configure(text=f"Wert: {float(value):.2f}")
    
    def check_detection_ready(self):
        """Prüft, ob Detection gestartet werden kann"""
        model_ready = bool(self.model_path_var.get())
        yaml_ready = bool(self.yaml_path_var.get()) and bool(self.class_names)
        
        if hasattr(self, 'detection_var'):
            detection_check = None
            # Widget finden
            for widget in self.root.winfo_children():
                if hasattr(widget, 'winfo_children'):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Checkbutton):
                            detection_check = child
                            break
            
            if detection_check:
                detection_check.configure(state="normal" if model_ready and yaml_ready else "disabled")
    
    def toggle_detection(self):
        """Objekterkennung ein-/ausschalten"""
        if self.detection_var.get():
            if not self.model_path_var.get() or not self.class_names:
                messagebox.showerror("Fehler", "Bitte Model und YAML-Datei auswählen!")
                self.detection_var.set(False)
                return
            
            try:
                # YOLO Model laden
                self.yolo_model = YOLO(self.model_path_var.get())
                self.detection_enabled = True
                self.detection_status_label.configure(text="Status: Aktiv", foreground="green")
                if hasattr(self, "streaming_detection_status_label"):
                    self.streaming_detection_status_label.configure(
                        text="Objekterkennung: Aktiv",
                        foreground="green",
                    )
                print(f"✅ YOLO Model geladen: {self.model_path_var.get()}")
                
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler beim Laden des Models: {e}")
                self.detection_var.set(False)
                return
        else:
            self.detection_enabled = False
            self.yolo_model = None
            self.detection_status_label.configure(text="Status: Inaktiv", foreground="red")
            if hasattr(self, "streaming_detection_status_label"):
                self.streaming_detection_status_label.configure(
                    text="Objekterkennung: Inaktiv",
                    foreground="orange",
                )
            print("❌ Objekterkennung deaktiviert")

        # Persist state
        self.save_detection_settings()
    
    def save_detection_settings(self, show_message: bool = True):
        """Speichert aktuelle Detection-Einstellungen"""
        try:
            settings = {
                'model_path': self.model_path_var.get(),
                'yaml_path': self.yaml_path_var.get(),
                'motion_threshold': self.motion_var.get(),
                'iou_threshold': self.iou_var.get(),
                'class_thresholds': {},
                'detection_enabled': self.detection_var.get()
            }
            
            # Klassen-Thresholds speichern
            for class_id, threshold_var in self.class_threshold_vars.items():
                settings['class_thresholds'][str(class_id)] = threshold_var.get()
            
            with open(self.detection_settings_file, 'w') as f:
                json.dump(settings, f, indent=4)

            self.project_manager.update_live_detection_settings(settings)

            if show_message:
                messagebox.showinfo("Erfolg", "Einstellungen gespeichert!")
                print(f"✅ Detection-Einstellungen gespeichert: {self.detection_settings_file}")
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Speichern: {e}")
    
    def load_detection_settings(self):
        """Lädt Detection-Einstellungen aus Datei"""
        try:
            settings = {}
            if Path(self.detection_settings_file).exists():
                with open(self.detection_settings_file, 'r') as f:
                    settings = json.load(f)

            pm_live = self.project_manager.get_live_detection_settings()
            # Only overwrite path fields if a valid value is stored in the project
            if pm_live.get("model_path"):
                settings["model_path"] = pm_live["model_path"]
            if pm_live.get("yaml_path"):
                settings["yaml_path"] = pm_live["yaml_path"]
            for key in ["motion_threshold", "iou_threshold", "class_thresholds", "detection_enabled", "enabled"]:
                if key in pm_live:
                    target_key = "enabled" if key in ["detection_enabled", "enabled"] else key
                    settings[target_key] = pm_live[key]

            print(f"✅ Detection-Einstellungen geladen: {self.detection_settings_file}")
            return settings

        except Exception as e:
            print(f"⚠️ Fehler beim Laden der Detection-Einstellungen: {e}")
            return {}
    
    def load_detection_settings_manual(self):
        """Lädt gespeicherte Einstellungen manuell"""
        self.detection_settings = self.load_detection_settings()
        self.project_manager.update_live_detection_settings(self.detection_settings)
        self.load_saved_paths()

        if hasattr(self, 'detection_var'):
            self.detection_var.set(self.detection_settings.get('enabled', False))
        
        # Threshold-Werte aktualisieren
        if hasattr(self, 'motion_var'):
            self.motion_var.set(self.detection_settings.get('motion_threshold', 110))
        if hasattr(self, 'iou_var'):
            self.iou_var.set(self.detection_settings.get('iou_threshold', 0.45))
        
        # Klassen-Thresholds aktualisieren
        if hasattr(self, 'class_threshold_vars'):
            for class_id, threshold_var in self.class_threshold_vars.items():
                saved_threshold = self.detection_settings.get('class_thresholds', {}).get(str(class_id), 0.7)
                threshold_var.set(saved_threshold)
                if class_id in self.class_threshold_labels:
                    self.class_threshold_labels[class_id].configure(text=f"{saved_threshold:.2f}")
        
        messagebox.showinfo("Erfolg", "Einstellungen geladen!")
    
    def load_saved_paths(self):
        """Lädt gespeicherte Pfade"""
        if hasattr(self, 'model_path_var') and 'model_path' in self.detection_settings:
            model_path = self.detection_settings['model_path']
            if Path(model_path).exists():
                self.model_path_var.set(model_path)
        
        if hasattr(self, 'yaml_path_var') and 'yaml_path' in self.detection_settings:
            yaml_path = self.detection_settings['yaml_path']
            if Path(yaml_path).exists():
                self.yaml_path_var.set(yaml_path)
                # YAML neu laden um Klassen zu erhalten
                try:
                    with open(yaml_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    self.class_names = data.get('names', {})
                    if isinstance(self.class_names, list):
                        self.class_names = {i: name for i, name in enumerate(self.class_names)}
                    
                    if hasattr(self, 'create_class_threshold_widgets'):
                        self.create_class_threshold_widgets()
                except Exception as e:
                    print(f"⚠️ Fehler beim Auto-Laden der YAML: {e}")
        
        self.check_detection_ready()
    
    def process_detection_on_frame(self, frame):
        """Führt YOLO-Detection auf Frame durch"""
        if not self.detection_enabled or not self.yolo_model:
            return frame
        
        try:
            # Motion Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            
            if self.prev_gray is None:
                self.prev_gray = gray
                return frame
            
            # Motion berechnen
            diff = cv2.absdiff(gray, self.prev_gray)
            max_diff = np.max(diff)
            self.prev_gray = gray
            
            # Static Check
            self.is_static = max_diff < self.motion_threshold
            
            # Status aktualisieren
            motion_status = "Statisch" if self.is_static else f"Bewegung ({max_diff:.1f})"
            self.motion_status_label.configure(text=f"Motion: {motion_status}")
            if hasattr(self, "streaming_motion_status_label"):
                self.streaming_motion_status_label.configure(
                    text=f"Motion: {motion_status}"
                )
            
            # Nur bei statischen Bildern Detection durchführen
            if not self.is_static:
                return frame
            
            # YOLO Detection durchführen
            results = self.yolo_model(
                frame,
                conf=0.1,  # Niedrige Basis-Confidence, filtern später
                iou=self.iou_var.get()
            )[0]
            
            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                self.detection_stats_label.configure(text="Erkennungen: 0")
                return frame
            
            # Klassen-spezifische Filterung
            cls_array = boxes.cls.cpu().numpy()
            conf_array = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            # Filter mit Klassen-spezifischen Thresholds
            valid_detections = np.zeros_like(cls_array, dtype=bool)
            for class_id, threshold_var in self.class_threshold_vars.items():
                class_mask = (cls_array == class_id) & (conf_array >= threshold_var.get())
                valid_detections |= class_mask
            
            # Gefilterte Detections
            cls_array = cls_array[valid_detections]
            conf_array = conf_array[valid_detections]
            xyxy = xyxy[valid_detections]
            
            # Bounding Boxes zeichnen
            annotated_frame = frame.copy()
            detection_count = 0
            
            for i in range(len(cls_array)):
                x1, y1, x2, y2 = map(int, xyxy[i])
                cls = int(cls_array[i])
                conf = conf_array[i]
                
                # Farbe basierend auf Klasse
                if cls == 0:
                    color = (20, 255, 57)  # Neon Grün
                elif cls == 1:
                    color = (0, 0, 255)    # Rot
                else:
                    color = (238, 130, 238)  # Violett
                
                # Box und Label zeichnen
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                class_name = self.class_names.get(cls, f"Klasse_{cls}")
                label = f"{class_name} {conf:.2f}"
                
                # Label-Hintergrund
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), 
                             (x1 + text_width, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                detection_count += 1
            
            # Statistiken aktualisieren
            class_counts = {}
            for class_id in self.class_names.keys():
                count = np.sum(cls_array == class_id)
                if count > 0:
                    class_counts[self.class_names[class_id]] = count
            
            stats_text = f"Erkennungen: {detection_count}"
            if class_counts:
                class_info = ", ".join([f"{name}: {count}" for name, count in class_counts.items()])
                stats_text += f" | {class_info}"
            
            self.detection_stats_label.configure(text=stats_text)
            
            return annotated_frame
            
        except Exception as e:
            print(f"❌ Detection-Fehler: {e}")
            return frame


def main(settings_dir=".", show_detection=False):
    """Hauptfunktion"""
    root = tk.Tk()
    app = IDSNXTCameraApp(root, settings_dir=settings_dir)

    if show_detection and hasattr(app, "detection_tab"):
        try:
            app.notebook.select(app.detection_tab)
        except Exception:
            pass
    # Programm beenden
    def on_closing():
        if app.streaming_active:
            app.stop_streaming()
        if app.monitor_active:
            app.stop_monitoring()
        app.save_all_settings()  # Finale Speicherung beim Schließen
        app.save_all_settings()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IDS NXT Camera App")
    parser.add_argument("settings_dir", nargs="?", default=".", help="Projektverzeichnis")
    parser.add_argument("--show-detection", action="store_true", help="Detection-Tab beim Start anzeigen")
    args = parser.parse_args()

    main(args.settings_dir, args.show_detection)