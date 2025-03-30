"""Parameter information buttons for the training settings."""

from PyQt6.QtWidgets import QToolButton, QMessageBox
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon, QFont, QColor
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
        # Create a message box with improved layout
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Parameter Information")
        msg_box.setText(self.info_text)
        msg_box.setIcon(QMessageBox.Icon.Information)
        
        # Style the message box
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #FFFFFF;
            }
            QMessageBox QLabel {
                color: #333333;
                font-size: 12px;
                min-width: 800px;  /* Make this much wider */
            }
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
        """)
        
        # Fix layout to give more space to text and less to icon
        # Set the total width of the dialog
        msg_box.setMinimumWidth(900)
        
        # Execute the dialog
        msg_box.exec()