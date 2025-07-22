import os
import cv2
import torch
import json
import uuid
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import numpy as np

# Konfigurasi folder
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Inisialisasi aplikasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load model YOLOv8
model = YOLO('best.pt')
class_names = model.names
colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype='uint8')

# Fungsi untuk validasi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Halaman utama (upload form)
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint untuk proses upload dan deteksi
@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Tidak ada file yang dipilih')
            return redirect(url_for('index'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('Tidak ada file yang dipilih')
            return redirect(url_for('index'))
        
        if file and allowed_file(file.filename):
            filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Proses deteksi
            result_filename = process_image(filepath, filename)
            
            # Redirect ke hasil
            return redirect(url_for('result', filename=filename, result=result_filename))
        
        flash('Format file tidak diizinkan. Gunakan PNG, JPG, atau JPEG.')
        return redirect(url_for('index'))
    
    # Kalau ada yang akses /detect langsung via GET, kembalikan ke index
    return redirect(url_for('index'))

# Fungsi untuk proses deteksi dan simpan metadata JSON (VERSI BARU DENGAN RESIZE)
def process_image(image_path, original_filename):
    image = cv2.imread(image_path)
    
    # --- TAMBAHKAN KODE RESIZE DI SINI ---
    target_width = 640
    (h, w) = image.shape[:2] # Ambil tinggi dan lebar asli
    
    # Hitung rasio aspek dan tentukan dimensi baru
    ratio = target_width / float(w)
    target_height = int(h * ratio)
    new_dim = (target_width, target_height)
    
    # Resize gambar dengan mempertahankan rasio aspek
    resized_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
    # ------------------------------------

    # --- GUNAKAN GAMBAR YANG SUDAH DI-RESIZE UNTUK DETEKSI ---
    results = model(resized_image)

    # Simpan gambar yang sudah di-resize ke folder hasil (tanpa anotasi)
    # Ini penting agar koordinat bounding box sesuai dengan gambar yang ditampilkan
    result_path = os.path.join(app.config['RESULT_FOLDER'], original_filename)
    cv2.imwrite(result_path, resized_image) # Simpan 'resized_image', bukan 'image' asli

    # Simpan metadata bounding box
    metadata = []
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # Pastikan class_names dan colors sudah didefinisikan secara global
                label = f"{class_names[cls]} {conf:.2f}"
                # Pastikan 'colors' didefinisikan dan dapat diakses
                color_index = cls % len(colors)
                color = colors[color_index]
                metadata.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'label': label,
                    'color': f'rgb({color[0]},{color[1]},{color[2]})'
                })

    # Simpan file JSON
    json_filename = f"{original_filename}.json"
    json_path = os.path.join(app.config['RESULT_FOLDER'], json_filename)
    with open(json_path, 'w') as f:
        json.dump(metadata, f)

# Halaman hasil deteksi
@app.route('/result/<filename>')
def result(filename):
    original = url_for('static', filename=f'results/{filename}')
    return render_template('result.html', original=original, original_filename=filename)

# Endpoint untuk melayani file JSON bounding box
@app.route('/results_data/<filename>')
def results_data(filename):
    json_path = os.path.join(app.config['RESULT_FOLDER'], f"{filename}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify([]), 404

# Jalankan aplikasi
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    app.run(debug=True)
