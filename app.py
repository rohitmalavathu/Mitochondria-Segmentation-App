from flask import Flask, render_template, request, jsonify
import os
import time
import glob
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def cleanup_old_uploads():
    """Clean up uploaded files older than 1 hour"""
    try:
        current_time = time.time()
        upload_dir = app.config['UPLOAD_FOLDER']
        
        # Get all files in uploads directory
        files = glob.glob(os.path.join(upload_dir, '*'))
        
        for file_path in files:
            if os.path.isfile(file_path):
                # Check if file is older than 1 hour (3600 seconds)
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > 3600:  # 1 hour
                    os.remove(file_path)
                    print(f"Cleaned up old file: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Clean up old files before processing new upload
    cleanup_old_uploads()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and process the image
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': 'Invalid image file'}), 400
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get image dimensions
            height, width = image_rgb.shape[:2]
            
            # Return success with file info and image dimensions
            return jsonify({
                'success': True, 
                'filename': filename,
                'width': width,
                'height': height,
                'message': 'File uploaded successfully'
            })
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/process', methods=['POST'])
def process_boxes():
    # Clean up old files before processing
    cleanup_old_uploads()
    
    try:
        data = request.get_json()
        boxes = data.get('boxes', [])
        scale_ratio = data.get('scale_ratio')
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Load the image for processing
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Could not load image'}), 400
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process each box
        results = []
        for i, box in enumerate(boxes):
            if isinstance(box, dict):
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            else:
                x1, y1, x2, y2 = box
            
            # Ensure coordinates are in correct order
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            
            # Crop the image to the box region
            cropped_image = image_rgb[y1:y2, x1:x2]
            
            if cropped_image.size == 0:
                continue
            
            # Simple segmentation using thresholding (mock for now)
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
            _, binary_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate area
            area_pixels = np.sum(binary_mask > 0)
            area_nm2 = None
            if scale_ratio:
                area_nm2 = area_pixels / (scale_ratio ** 2)
            
            # Convert contours to original image coordinates
            contour_points = []
            for contour in contours:
                if contour is not None and len(contour) > 0:
                    adjusted_contour = contour.reshape(-1, 2) + [x1, y1]
                    contour_points.append(adjusted_contour.tolist())
            
            results.append({
                'box_id': i,
                'area_pixels': int(area_pixels),
                'area_nm2': float(area_nm2) if area_nm2 else None,
                'contours': contour_points,
                'box_coords': [x1, y1, x2, y2]
            })
        
        # Clean up uploaded file after processing
        try:
            os.remove(filepath)
            print(f"Cleaned up uploaded file: {filename}")
        except Exception as cleanup_error:
            print(f"Warning: Could not delete uploaded file {filename}: {cleanup_error}")
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
