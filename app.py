from flask import Flask, render_template, request, jsonify, os
import cv2
import numpy as np
import base64
from datetime import datetime
from fpdf import FPDF

app = Flask(__name__)

RESULT_FOLDER = 'static/results'
os.makedirs(RESULT_FOLDER, exist_ok=True)

def classify_crystal_violet(image, rows=8, cols=2):
    # Resize to ensure clean division for the grid
    image = cv2.resize(image, (cols * 150, rows * 100))
    output_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    cell_h, cell_w = h // rows, w // cols

    results = []
    for i in range(rows):
        for j in range(cols):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = gray[y1:y2, x1:x2]
            mean_val = np.mean(cell)

            # Intensity Logic 
            if mean_val < 50:
                level, uv = "Very High Agglutination", "30-40 min"
            elif mean_val < 100:
                level, uv = "High Agglutination", "20-30 min"
            elif mean_val < 150:
                level, uv = "Moderate Agglutination", "10-20 min"
            elif mean_val < 200:
                level, uv = "Low Agglutination", "5-10 min"
            else:
                level, uv = "No Agglutination", "Not Required"

            well_num = i * cols + j + 1
            results.append({"well": well_num, "level": level, "uv": uv, "val": int(mean_val)})

            # Draw UI labels
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.putText(output_image, f"W{well_num}", (x1 + 5, y1 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    img_filename = f"result_{timestamp}.png"
    pdf_filename = f"report_{timestamp}.pdf"
    
    img_path = os.path.join(RESULT_FOLDER, img_filename)
    pdf_path = os.path.join(RESULT_FOLDER, pdf_filename)
    
    cv2.imwrite(img_path, output_image)
    
    # Generate PDF with Image
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(190, 10, txt="Crystal Violet Agglutination Report", ln=True, align='C')
    pdf.ln(5)
    
    # Insert the processed image into PDF
    pdf.image(img_path, x=10, y=None, w=100)
    pdf.ln(5)
    
    pdf.set_font("Arial", size=10)
    for res in results:
        text = f"Well {res['well']}: {res['level']} | Intensity: {res['val']} | UV: {res['uv']}"
        pdf.cell(190, 8, txt=text, ln=True)
    
    pdf.output(pdf_path)

    summary = f"Analysis Complete: {rows}x{cols} grid processed."
    return summary, f"/static/results/{img_filename}", f"/static/results/{pdf_filename}"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        img_b64 = data.get('image').split(',')[1]
        # Default to 8x2 if not specified 
        rows = int(data.get('rows', 8))
        cols = int(data.get('cols', 2))
        
        img_data = base64.b64decode(img_b64)
        npimg = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        res_text, img_url, pdf_url = classify_crystal_violet(img, rows, cols)
        return jsonify({'result': res_text, 'image_url': img_url, 'pdf_url': pdf_url})
    except Exception as e:
        return jsonify({'result': f"Error: {str(e)}"}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Fixed for Render: listen on 0.0.0.0 and dynamic PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
