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
        self.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        
        # Make the button more visible on light background
        self.setStyleSheet("""
            QToolButton {
                color: #1976D2;
                background-color: #E3F2FD;
                border: 1px solid #1976D2;
                border-radius: 12px;
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
        
        self.setFixedSize(24, 24)
        self.setToolTip("Click for more information")
        self.clicked.connect(self.show_info)

    def show_info(self):
        """Show information dialog."""
        # Mimic the style of dashboard help dialogs for consistent layout
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Parameter Information")
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setTextFormat(Qt.TextFormat.RichText)

        # Convert plain newlines to paragraphs for better readability
        paragraphs = [p.strip() for p in self.info_text.split("\n") if p.strip()]
        formatted = "<br><br>".join(paragraphs)
        msg_box.setText(formatted)

        # Ensure the label within the message box wraps the text
        label = msg_box.findChild(QLabel, "qt_msgbox_label")
        if label is not None:
            label.setWordWrap(True)