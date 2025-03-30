"""Detailed information about training metrics."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QLabel, QTextBrowser
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
import markdown

def get_metrics_info_html():
    """Return formatted HTML with detailed metrics information."""
    metrics_info = """
# YOLO Training Metrics Guide für industrielle Fehlerinspektion

## Übersicht

Diese Metriken helfen Ihnen zu verstehen, wie gut Ihr Modell Fehler in Spritzgussteilen erkennt. Die folgenden Erklärungen sind sowohl für Anfänger als auch für erfahrene Benutzer gedacht.

## Verlustfunktionen (Loss Functions)

### Box Loss

**Was ist das?** 
Dieser Wert zeigt, wie genau das Modell die Position und Größe von Fehlerregionen (Bounding Boxes) vorhersagt.

**Für Anfänger:**
- Niedriger Wert = Das Modell platziert die Bounding Boxes sehr genau um die Fehler
- Hoher Wert = Das Modell hat Schwierigkeiten, Fehler präzise zu lokalisieren

**Für Fortgeschrittene:**
- Der Box Loss bewertet die Qualität der Bounding Box Regression mittels CIOU-Loss
- Er berücksichtigt Überlappung, Mittelpunktdistanz, Aspektverhältnis und Größe
- Typische Werte:
  - Ausgezeichnet: < 0.05
  - Gut: 0.05 - 0.1
  - Akzeptabel: 0.1 - 0.3
  - Verbesserungsbedürftig: > 0.3

**Industrielle Relevanz:**
In der Fehlerinspektion von Spritzgussteilen ist die präzise Lokalisierung wichtig, um:
- Die genaue Position von Fehlern zu dokumentieren
- Prozessanalysen zu ermöglichen
- Qualitätskriterien zu definieren

### Class Loss

**Was ist das?**
Zeigt, wie gut das Modell zwischen verschiedenen Fehlertypen (z.B. Grate, Lunker, Risse) unterscheiden kann.

**Für Anfänger:**
- Niedriger Wert = Das Modell erkennt Fehlertypen zuverlässig
- Hoher Wert = Das Modell verwechselt Fehlertypen häufig

**Für Fortgeschrittene:**
- Basiert auf Categorical Cross-Entropy mit Label-Smoothing
- Misst die Diskrepanz zwischen vorhergesagten und tatsächlichen Klassen
- Typische Werte:
  - Ausgezeichnet: < 0.02
  - Gut: 0.02 - 0.05
  - Akzeptabel: 0.05 - 0.15
  - Verbesserungsbedürftig: > 0.15

**Industrielle Relevanz:**
Die korrekte Klassifizierung von Fehlertypen ist entscheidend für:
- Automatisierte Sortier- und Ausschusssysteme
- Nachvollziehbare Qualitätsdokumentation
- Gezielte Prozessoptimierung nach Fehlertyp

### DFL Loss (Distribution Focal Loss)

**Was ist das?**
Ein spezieller Loss für präzise Kantenlokalisierung bei Bounding Boxes.

**Für Anfänger:**
- Niedriger Wert = Modell definiert Fehlerkanten sehr genau
- Hoher Wert = Ungenaue Fehlerbegrenzung

**Für Fortgeschrittene:**
- Verbessert die Präzision bei der Bounding Box Regression
- Konzentriert sich auf die Verteilung der Koordinaten statt einzelner Punktschätzungen
- Typische Werte:
  - Ausgezeichnet: < 0.03
  - Gut: 0.03 - 0.08
  - Akzeptabel: 0.08 - 0.15
  - Verbesserungsbedürftig: > 0.15

**Industrielle Relevanz:**
Bei der Prüfung von Spritzgussteilen wichtig für:
- Genaue Vermessung von Fehlergröße
- Bestimmung, ob Fehler innerhalb von Toleranzgrenzen liegen
- Exakte Dokumentation für Qualitätsberichte

## Leistungsmetriken (Performance Metrics)

### Precision

**Was ist das?**
Der Prozentsatz der korrekt erkannten Fehler unter allen erkannten Fehlern.

**Für Anfänger:**
- Hoher Wert = Wenige falsch positive Fehlererkennungen (kaum "Fehlalarme")
- Niedriger Wert = Viele Fehlalarme

**Für Fortgeschrittene:**
- Berechnung: TP / (TP + FP)
- TP = True Positives (korrekt erkannte Fehler)
- FP = False Positives (fälschlicherweise erkannte "Fehler")
- Typische Werte:
  - Ausgezeichnet: > 95%
  - Gut: 90% - 95%
  - Akzeptabel: 80% - 90%
  - Verbesserungsbedürftig: < 80%

**Industrielle Relevanz:**
Bei der Fehlerinspektion besonders wichtig, um:
- Unnötige Ausschussproduktion zu vermeiden
- Operatorvertrauen zu stärken
- Produktionskosten zu optimieren

### Recall

**Was ist das?**
Der Prozentsatz der erkannten Fehler im Verhältnis zu allen tatsächlichen Fehlern.

**Für Anfänger:**
- Hoher Wert = Fast alle tatsächlichen Fehler werden gefunden
- Niedriger Wert = Viele Fehler werden übersehen

**Für Fortgeschrittene:**
- Berechnung: TP / (TP + FN)
- TP = True Positives (korrekt erkannte Fehler)
- FN = False Negatives (übersehene Fehler)
- Typische Werte:
  - Ausgezeichnet: > 95%
  - Gut: 90% - 95%
  - Akzeptabel: 80% - 90%
  - Verbesserungsbedürftig: < 80%

**Industrielle Relevanz:**
Bei der Qualitätssicherung entscheidend für:
- Minimierung von fehlerhaften Teilen, die zum Kunden gelangen
- Compliance mit Qualitätsstandards
- Risikominimierung bei sicherheitskritischen Teilen

### mAP50 (Mean Average Precision bei IoU=0.50)

**Was ist das?**
Die durchschnittliche Genauigkeit der Fehlererkennung bei 50% Überlappung.

**Für Anfänger:**
- Hoher Wert = Modell erkennt Fehler zuverlässig mit ausreichender Genauigkeit
- Niedriger Wert = Unzuverlässige Fehlererkennung

**Für Fortgeschrittene:**
- Mittlerer AP-Wert über alle Fehlerklassen bei einem IoU-Schwellenwert von 0.5
- IoU (Intersection over Union) = Überlappungsgrad zwischen vorhergesagter und tatsächlicher Box
- Typische Werte:
  - Ausgezeichnet: > 95%
  - Gut: 90% - 95%
  - Akzeptabel: 80% - 90%
  - Verbesserungsbedürftig: < 80%

**Industrielle Relevanz:**
Bietet eine gute Balance für:
- Zuverlässige Fehlererkennung in Spritzgussteilen
- Kompromiss zwischen Genauigkeit und Rechenaufwand
- Robustheit gegenüber leichten Variationen in der Fehlerdarstellung

### mAP50-95 (Durchschnitt über mehrere IoU-Schwellen)

**Was ist das?**
Die durchschnittliche Genauigkeit über verschiedene Überlappungsgrade (50%-95%).

**Für Anfänger:**
- Hoher Wert = Modell lokalisiert Fehler sehr präzise
- Niedriger Wert = Modell erkennt Fehler, aber mit ungenauen Grenzen

**Für Fortgeschrittene:**
- Mittlerer AP-Wert über IoU-Schwellen von 0.5 bis 0.95 (in 0.05-Schritten)
- Höherer Standard als mAP50, bewertet die Lokalisierungsgenauigkeit stärker
- Typische Werte:
  - Ausgezeichnet: > 80%
  - Gut: 70% - 80%
  - Akzeptabel: 60% - 70%
  - Verbesserungsbedürftig: < 60%

**Industrielle Relevanz:**
Besonders wichtig bei:
- Präzisionsteilen mit engen Toleranzgrenzen
- Automatisierten Messsystemen für Fehlerauswertung
- Detaillierter Fehleranalyse für Prozessoptimierung

## Trainingsparameter

### Learning Rate

**Was ist das?**
Die Schrittgröße bei der Anpassung der Modellgewichte während des Trainings.

**Für Anfänger:**
- Zu hoch = Training wird instabil
- Zu niedrig = Training dauert sehr lange
- Optimal = Lernfortschritt ohne Instabilität

**Für Fortgeschrittene:**
- Steuert die Geschwindigkeit der Parameteranpassung
- Mit Cosine-Scheduling wird sie im Trainingsverlauf reduziert
- Typische Werte:
  - Optimal: 0.001 - 0.005
  - Aggressiv: 0.005 - 0.01
  - Konservativ: 0.0005 - 0.001

**Industrielle Relevanz:**
Beeinflusst entscheidend:
- Trainingszeit bis zur Produktionsreife
- Endgültige Modellgenauigkeit
- Robustheit gegenüber Variationen in Fehlerbildern
"""

    return markdown.markdown(metrics_info)

def create_metrics_info_tab():
    """Create a tab with detailed metrics information."""
    
    tab = QWidget()
    layout = QVBoxLayout(tab)
    
    title = QLabel("Metriken-Information für industrielle Objekterkennung")
    title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
    title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(title)
    
    # Use QTextBrowser to display formatted text
    info_browser = QTextBrowser()
    info_browser.setOpenExternalLinks(True)
    info_browser.setHtml(get_metrics_info_html())
    # Make sure text is dark on white background
    info_browser.setStyleSheet("""
        QTextBrowser {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 14px;
            line-height: 1.5;
            color: #333333;
        }
        QTextBrowser p {
            color: #333333;
        }
        QTextBrowser li {
            color: #333333;
        }
        h1 {
            color: #1976D2;
            margin-bottom: 15px;
        }
        h2 {
            color: #2196F3;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        h3 {
            color: #0D47A1;
            margin-top: 15px;
        }
        strong {
            font-weight: bold;
            color: #333;
        }
        * {
            color: #333333;
        }
    """)
    
    layout.addWidget(info_browser)
    
    return tab