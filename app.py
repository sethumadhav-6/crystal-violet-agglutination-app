from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import cv2
import numpy as np
import base64
import os
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Directory to save result images
RESULT_FOLDER = 'static/results'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def classify_crystal_violet(image, grid_size=4):
    image = cv2.resize(image, (400, 400))  # Resize for 4x4 grid
    output_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    cell_h, cell_w = h // grid_size, w // grid_size

    results = []
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = gray[y1:y2, x1:x2]
            mean_val = np.mean(cell)

            if mean_val < 50:
                level = "Very High Agglutination"
                uv = "30-40 min"
            elif mean_val < 100:
                level = "High Agglutination"
                uv = "20-30 min"
            elif mean_val < 150:
                level = "Moderate Agglutination"
                uv = "10-20 min"
            elif mean_val < 200:
                level = "Low Agglutination"
                uv = "5-10 min"
            else:
                level = "No Agglutination"
                uv = "Not Required"

            well_num = i * grid_size + j + 1
            results.append((well_num, level, uv, mean_val, (x1, y1, x2, y2)))

            label = f"W{well_num}"
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.putText(output_image, label, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Create result text summary
    summary = "Agglutination Summary:\n"
    for well_num, level, uv, intensity, _ in results:
        summary += f"Well {well_num}: {level} (Intensity={int(intensity)}) | UV: {uv}\n"

    # Save image
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"result_{timestamp}.png"
    filepath = os.path.join(RESULT_FOLDER, filename)
    cv2.imwrite(filepath, output_image)

    return summary.strip(), f"/static/results/{filename}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json().get('image')
        if not data:
            return jsonify({'result': 'No image data received'}), 400

        encoded_data = data.split(',')[1]
        img_data = base64.b64decode(encoded_data)
        npimg = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'result': 'Image decoding failed'}), 400

        result, image_path = classify_crystal_violet(img)
        return jsonify({'result': result, 'image_url': image_path})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'result': f'Error analyzing image: {str(e)}'}), 500


@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return redirect(url_for('index'))

        file = request.files['file']
        if file.filename == '':
            return redirect(url_for('index'))

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            result, image_path = classify_crystal_violet(img)
            return render_template('index.html', uploaded_result=result, uploaded_image=image_path)

    except Exception as e:
        print(f"Upload error: {e}")
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
