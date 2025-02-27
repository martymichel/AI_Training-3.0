from flask import Flask, render_template, request, jsonify
from pathlib import Path
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import time

app = Flask(__name__, static_folder='static')

# Farben für Bounding-Boxen
BOX_COLORS = {
    0: "red",
    1: "black",
    2: "blue",
    3: "green",
    4: "yellow"
}

# Cache für Dataset
dataset_cache = {
    'image_paths': None,
    'stats': None
}

def find_matching_label(image_path):
    """Passende Label-Datei für ein Bild finden."""
    image_name = image_path.stem
    parent_dir = image_path.parent
    
    label_path = image_path.with_suffix('.txt')
    if label_path.exists():
        return label_path
        
    if 'images' in str(parent_dir).lower():
        possible_label_dir = str(parent_dir).lower().replace('images', 'labels')
        label_path = Path(possible_label_dir) / f"{image_name}.txt"
        if label_path.exists():
            return label_path
            
    return None

def count_unique_labels(label_path):
    """Einzigartige Labels in einer Label-Datei zählen."""
    unique_labels = set()
    try:
        with open(label_path, 'r') as f:
            for line in f:
                try:
                    class_id = int(line.strip().split()[0])
                    unique_labels.add(class_id)
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass
    return unique_labels

def process_image(img, label_path):
    """Bild mit Bounding-Boxen annotieren."""
    if label_path:
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    try:
                        class_id, x_center, y_center, width, height = map(
                            float, line.strip().split())
                        
                        x_center_px = x_center * img_width
                        y_center_px = y_center * img_height
                        width_px = width * img_width
                        height_px = height * img_height
                        
                        left = x_center_px - width_px / 2
                        top = y_center_px - height_px / 2
                        right = x_center_px + width_px / 2
                        bottom = y_center_px + height_px / 2
                        
                        box_color = BOX_COLORS.get(int(class_id), "white")
                        draw.rectangle([left, top, right, bottom], outline=box_color, width=3)
                    except Exception as e:
                        print(f"Fehler beim Verarbeiten der Label-Zeile: {e}")
                        continue
        except Exception as e:
            print(f"Fehler beim Lesen der Label-Datei: {e}")

def image_to_base64(img):
    """PIL-Bild in Base64 umwandeln."""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def get_image_files(dataset_path):
    """Alle Bilddateien im Dataset-Verzeichnis abrufen."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_paths = []
    dataset_path = Path(dataset_path)
    
    try:
        abs_path = dataset_path.resolve()
        for file_path in abs_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_paths.append(file_path)
    except Exception as e:
        print(f"Fehler beim Scannen des Verzeichnisses: {e}")
    
    return image_paths

def analyze_dataset(dataset_path):
    """Dataset analysieren."""
    stats = {
        'total_images': 0,
        'background_images': 0,
        'total_labels': 0
    }
    
    try:
        image_paths = get_image_files(dataset_path)
        all_label_files = set()
        stats['total_images'] = len(image_paths)
        
        for img_path in image_paths:
            label_path = find_matching_label(img_path)
            if label_path:
                all_label_files.add(label_path)
            else:
                stats['background_images'] += 1
        
        unique_labels = set()
        for label_path in all_label_files:
            file_labels = count_unique_labels(label_path)
            unique_labels.update(file_labels)
        
        stats['total_labels'] = len(unique_labels)
        stats['annotated_images'] = stats['total_images'] - stats['background_images']
        
        return image_paths, stats
    except Exception as e:
        print(f"Fehler beim Verarbeiten des Datasets: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_dataset', methods=['POST'])
def analyze_dataset_route():
    dataset_path = request.form['dataset_path']
    
    try:
        dataset_path = Path(dataset_path).resolve()
        if not dataset_path.exists():
            return jsonify({'error': f"Pfad existiert nicht: {dataset_path}"}), 400
        if not dataset_path.is_dir():
            return jsonify({'error': f"Pfad muss ein Verzeichnis sein: {dataset_path}"}), 400
        
        start_time = time.time()
        image_paths, stats = analyze_dataset(dataset_path)
        analysis_time = time.time() - start_time
        
        if not image_paths:
            return jsonify({'error': "Keine Bilder im Dataset gefunden"}), 400
            
        dataset_cache['image_paths'] = image_paths
        dataset_cache['stats'] = stats
        
        return jsonify({
            'stats': stats,
            'total_images': len(image_paths),
            'analysis_time': round(analysis_time, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_page', methods=['POST'])
def get_page():
    if not dataset_cache['image_paths']:
        return jsonify({'error': 'Kein Dataset geladen'}), 400
        
    try:
        page = int(request.form.get('page', 1))
        start_idx = (page - 1) * 9
        end_idx = start_idx + 9
        start_time = time.time()
        
        page_images = []
        for img_path in dataset_cache['image_paths'][start_idx:end_idx]:
            try:
                with Image.open(img_path) as img:
                    label_path = find_matching_label(img_path)
                    if label_path:
                        process_image(img, label_path)
                    page_images.append({
                        'path': str(img_path),
                        'data': image_to_base64(img)
                    })
            except Exception as e:
                print(f"Fehler beim Verarbeiten des Bildes {img_path}: {e}")
                continue
                
        load_time = time.time() - start_time
        
        return render_template('gallery.html', 
                            images=page_images,
                            stats=dataset_cache['stats'],
                            current_page=page,
                            total_pages=(len(dataset_cache['image_paths']) + 8) // 9,
                            load_time=round(load_time, 2))
    except Exception as e:
        return render_template('gallery.html', error=f"Fehler beim Laden der Seite: {e}")

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5003, debug=True)
