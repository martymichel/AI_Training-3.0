# utils/labeling_utils.py

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Wandelt eine Bounding Box (x_min, y_min, x_max, y_max) in YOLO-Koordinaten 
    (x_center, y_center, width, height) um. Alle Werte werden relativ zur Bildgröße normiert.
    """
    x_min, y_min, x_max, y_max = bbox
    box_width = x_max - x_min
    box_height = y_max - y_min
    x_center = x_min + box_width / 2.0
    y_center = y_min + box_height / 2.0
    return x_center / img_width, y_center / img_height, box_width / img_width, box_height / img_height

def save_yolo_labels(file_path, annotations, img_width, img_height):
    """
    Speichert Annotationen im YOLO-Format in einer Textdatei.
    :param annotations: Liste von Tupeln (class_id, x_min, y_min, x_max, y_max)
    """
    with open(file_path, 'w') as f:
        for ann in annotations:
            class_id, x_min, y_min, x_max, y_max = ann
            x_center, y_center, box_width, box_height = convert_bbox_to_yolo(
                (x_min, y_min, x_max, y_max), img_width, img_height)
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

def save_classes_file(file_path, class_names):
    """
    Speichert die Klassennamen zeilenweise in einer Datei.
    """
    with open(file_path, 'w') as f:
        for name in class_names:
            f.write(name + "\n")
