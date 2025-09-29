"""Parameter information buttons for the training settings."""

from PyQt6.QtWidgets import QToolButton, QMessageBox, QLabel
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

class ParameterInfoButton(QToolButton):
    """Custom button for parameter information."""
    def __init__(self, info_text, parent=None):
        super().__init__(parent)
        self.info_text = info_text
        
        # Using standard icon but with dark colors
        self.setText("ℹ️")
        self.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        
        # Make the button more visible on light background
        self.setStyleSheet("""
            QToolButton {
                color: #1976D2;
                background-color: #E3F2FD;
                border: 1px solid #1976D2;
                border-radius: 10px;
                padding: 2px;
            }
            QToolButton:hover {
                background-color: #BBDEFB;
            }
            QToolButton:pressed {
                background-color: #1976D2;
                color: white;
            }
        """)
        
        self.setFixedSize(20, 20)
        self.setToolTip("Click for more information")
        self.clicked.connect(self.show_info)

    def show_info(self):
        """Show information dialog."""
        # Create a properly styled message box
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Parameter Information")
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setMinimumSize(500, 300)

        # Convert plain newlines to paragraphs for better readability
        paragraphs = [p.strip() for p in self.info_text.split("\n") if p.strip()]
        formatted = "<br><br>".join(paragraphs)
        msg_box.setText(formatted)
        
        # Style the message box
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QMessageBox QLabel {
                color: #333;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
        """)

        msg_box.exec()