import os
from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO(r"G:\OneDrive - Flex\3_ARBEIT ARBEIT ARBEIT\5.2_AI_Camera tests\M_inner_B_Reklamation_3AA01995\Train1\weights\best.pt")

# Export the model to NCNN format
model.export(format="ncnn")  # creates a 'best-ncnn' folder in the same directory as the model