import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import base64
from io import BytesIO
from PIL import Image
import json
from huggingface_hub import hf_hub_download
import time
import glob

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model
sam2_model = None
predictor = None
model_loaded = False

def load_model():
    """Load the SAM2 model and fine-tuned weights"""
    global sam2_model, predictor, model_loaded
    
    if model_loaded:
        return True
        
    try:
        print("Loading SAM2 model...")
        FINE_TUNED_MODEL_WEIGHTS = hf_hub_download(repo_id="rohitmalavathu/SAM2FineTunedMito", filename="fine_tuned_sam2_2000.torch")
        sam2_checkpoint = hf_hub_download(repo_id="rohitmalavathu/SAM2FineTunedMito", filename="sam2_hiera_small.pt")
        model_cfg = "sam2_hiera_s.yaml"
        
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS, map_location=torch.device('cpu'), weights_only=False))
        
        model_loaded = True
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

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

def process_segmentation(image, boxes, scale_ratio=None):
    """Process segmentation for multiple boxes"""
    results = []
    
    for i, box in enumerate(boxes):
        # Handle both list and dict formats
        if isinstance(box, dict):
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        else:
            x1, y1, x2, y2 = box
        
        # Ensure coordinates are in correct order
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        
        # Crop the image
        cropped_image = image[y1:y2, x1:x2]
        original_height, original_width = cropped_image.shape[:2]
        
        # Resize for model input
        cropped_image_resized = cv2.resize(cropped_image, (256, 256))
        
        # Convert to 3-channel if grayscale
        if cropped_image_resized.ndim == 2:
            cropped_image_resized = np.stack([cropped_image_resized] * 3, axis=-1)
        
        # Process with SAM2
        with torch.no_grad():
            predictor.set_image(cropped_image_resized)
            masks, scores, logits = predictor.predict(
                point_coords=[[[128, 128]]],
                point_labels=[[1]]
            )
        
        # Process masks
        sorted_masks = masks[np.argsort(scores)][::-1]
        seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
        occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
        
        for j in range(sorted_masks.shape[0]):
            mask = sorted_masks[j]
            if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
                continue
            
            mask_bool = mask.astype(bool)
            mask_bool[occupancy_mask] = False
            seg_map[mask_bool] = j + 1
            occupancy_mask[mask_bool] = True
        
        # Smooth the segmentation
        seg_mask = gaussian_filter(seg_map.astype(float), sigma=2)
        smoothed_mask = (seg_mask > 0.5).astype(np.uint8)
        segmentation_resized = cv2.resize(smoothed_mask, (original_width, original_height))
        
        # Calculate area using the resized segmentation (actual cropped box size)
        area_pixels = np.sum(segmentation_resized)
        # Convert pixels to nm²: area_pixels / (pixels_per_nm)²
        area_nm2 = area_pixels / (scale_ratio ** 2) if scale_ratio else area_pixels
        
        # Find contours for outline
        segmentation_255 = segmentation_resized * 255
        contours, _ = cv2.findContours(segmentation_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert contours to list for JSON serialization
        # Adjust contours to be relative to the full image coordinates
        contour_points = []
        for contour in contours:
            if contour is not None and len(contour) > 0:
                # Adjust contour points to be relative to the full image
                adjusted_contour = contour.reshape(-1, 2) + [x1, y1]
                contour_points.append(adjusted_contour.tolist())
        
        print(f"Box {i}: Found {len(contour_points)} contours")
        for j, contour in enumerate(contour_points):
            print(f"  Contour {j}: {len(contour)} points")
            if len(contour) > 0:
                print(f"    First point: {contour[0]}")
                print(f"    Last point: {contour[-1]}")
        
        results.append({
            'box_id': i,
            'area_pixels': int(area_pixels),
            'area_nm2': float(area_nm2) if scale_ratio else None,
            'contours': contour_points,
            'mask_shape': smoothed_mask.shape,
            'box_coords': [x1, y1, x2, y2]  # Add original box coordinates for display
        })
    
    return results

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
            
            # Calculate scaling for display
            canvas_size = 1024
            scale_factor = min(canvas_size / width, canvas_size / height)
            scaled_width = int(width * scale_factor)
            scaled_height = int(height * scale_factor)
            
            # Resize image for display
            resized_image = cv2.resize(image_rgb, (scaled_width, scaled_height))
            
            # Convert to base64 for frontend
            pil_image = Image.fromarray(resized_image)
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Calculate offset to center image
            offset_x = (canvas_size - scaled_width) // 2
            offset_y = (canvas_size - scaled_height) // 2
            
            # Return success with file info and image dimensions
            return jsonify({
                'success': True, 
                'filename': filename,
                'width': width,
                'height': height,
                'image': f"data:image/png;base64,{img_str}",
                'scale_factor': scale_factor,
                'image_offset': {'x': offset_x, 'y': offset_y},
                'image_size': {'width': scaled_width, 'height': scaled_height},
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
        
        # Load model if not already loaded
        if not load_model():
            return jsonify({'error': 'Failed to load SAM2 model'}), 500
        
        # Process segmentation using the real function
        results = process_segmentation(image_rgb, boxes, scale_ratio)
        
        # Clean up uploaded file after processing
        try:
            os.remove(filepath)
            print(f"Cleaned up uploaded file: {filename}")
        except Exception as cleanup_error:
            print(f"Warning: Could not delete uploaded file {filename}: {cleanup_error}")
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get the status of the model loading"""
    load_model()
    return jsonify({
        'model_loaded': model_loaded,
        'model_available': predictor is not None,
        'error': model_loading_error
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
