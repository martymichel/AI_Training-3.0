Die Tools bilden eine vollständige End-to-End-Pipeline für die Entwicklung und Anwendung von Computer Vision-Modellen, speziell für die Objekterkennung mit YOLO:

# Datenerfassung

Beginnend mit der Kamera-App (Nr. 1), die das Aufnehmen von Bildern für die Erstellung eines Datensatzes ermöglicht.

# Datenaufbereitung und -annotierung

Das Labeling-Tool (Nr. 2) erlaubt das manuelle Markieren von Objekten mit Bounding-Boxen, was essentiell für das überwachte Lernen ist.

# Datensatzvergrösserung
Die Augmentierungs-App (Nr. 3) erweitert den Datensatz durch verschiedene Transformationen (Rotation, Spiegelung, etc.), was die Modellgeneralisierung verbessert.

# Datensatzvalidierung

Der Dataset-Viewer (Nr. 4) ermöglicht die Überprüfung der annotierten Bilder, was für die Qualitätssicherung wichtig ist.

# Datensatzaufteilung

Der Dataset-Splitter (Nr. 5) teilt die Daten in Trainings-, Validierungs- und Testsets auf - ein notwendiger Schritt für robustes ML-Training.

# Modelltraining

Der YOLO-Trainer (Nr. 6) konfiguriert und startet das eigentliche Training des neuronalen Netzwerks.

## Training fortsetzen

Im Trainer kann die Option **Resume** aktiviert werden, um ein zuvor gestartetes Training weiterzuführen. Das Programm sucht dann nach einer `weights/last.pt`-Datei im angegebenen Projekt und Experiment. Ist kein Checkpoint vorhanden, beginnt ein neues Training und es erscheint eine Warnung im Log.

# Trainingsüberwachung

Das Dashboard (zugänglich über Trainer) visualisiert Metriken und Fortschritt während des Trainings.

# Modellverifizierung

Das Verifikationstool (Nr. 7) testet das trainierte Modell gegen einen separaten Testdatensatz, um die Leistung zu bewerten.

# Modellanwendung

Die Live-Detektions-App (Nr. 8) wendet das trainierte Modell auf einen Kamera-Livestream an für Echtzeiterkennung.

Die Anwendungen sind logisch geordnet und bilden einen vollständigen ML-Workflow von der Datensammlung bis zur Bereitstellung. Jedes Tool erzeugt Ausgaben, die vom nächsten Tool in der Kette verwendet werden können, wodurch ein kohärenter, durchgehender Prozess entsteht.

Die Tools sind auch so gestaltet, dass sie sowohl für Einsteiger als auch für erfahrene Anwender zugänglich sind, mit ausführlichen Tooltips und Informationsdialogen zur Erklärung der verschiedenen Parameter und Optionen.