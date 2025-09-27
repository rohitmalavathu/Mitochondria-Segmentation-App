from flask import Flask, render_template, request, jsonify
import os
import time
import glob
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model
sam2_model = None
predictor = None
model_loaded = False
model_loading_error = None

def load_model():
    """Load the SAM2 model with fallback to mock mode"""
    global sam2_model, predictor, model_loaded, model_loading_error
    
    if model_loaded:
        return True
        
    try:
        print("Attempting to load SAM2 model...")
        
        # Try to import SAM2 components
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            print(f"SAM2 not available: {e}")
            model_loading_error = "SAM2 model not available - using mock mode"
            model_loaded = True
            return True
        
        # Try to load model weights
        try:
            # Use a smaller, more reliable model
            sam2_checkpoint = hf_hub_download(repo_id="facebook/sam2-hiera_small", filename="sam2_hiera_small.pt")
            model_cfg = "sam2_hiera_s.yaml"
            
            sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")
            predictor = SAM2ImagePredictor(sam2_model)
            
            model_loaded = True
            print("SAM2 model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading SAM2 model: {e}")
            model_loading_error = f"SAM2 model loading failed: {str(e)} - using mock mode"
            model_loaded = True
            return True
            
    except Exception as e:
        print(f"Unexpected error during model loading: {e}")
        model_loading_error = f"Model loading error: {str(e)} - using mock mode"
        model_loaded = True
        return True

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
    """Process segmentation for multiple boxes with real SAM2 or fallback to mock"""
    results = []
    
    # Load model if not already loaded
    load_model()
    
    for i, box in enumerate(boxes):
        # Handle both list and dict formats
        if isinstance(box, dict):
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        else:
            x1, y1, x2, y2 = box
        
        # Ensure coordinates are in correct order
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        
        print(f"Box {i}: Processing box at ({x1}, {y1}, {x2}, {y2})")
        
        # Crop the image to the box region
        cropped_image = image[y1:y2, x1:x2]
        original_height, original_width = cropped_image.shape[:2]
        
        if cropped_image.size == 0:
            continue
        
        # Try real SAM2 segmentation if available
        if predictor is not None:
            try:
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
                
                # Calculate area
                area_pixels = np.sum(segmentation_resized)
                area_nm2 = None
                if scale_ratio:
                    area_nm2 = area_pixels / (scale_ratio ** 2)
                
                # Find contours
                segmentation_255 = segmentation_resized * 255
                contours, _ = cv2.findContours(segmentation_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Convert contours to original image coordinates
                contour_points = []
                for contour in contours:
                    if contour is not None and len(contour) > 0:
                        adjusted_contour = contour.reshape(-1, 2) + [x1, y1]
                        contour_points.append(adjusted_contour.tolist())
                
                print(f"Box {i}: Real SAM2 segmentation - Area: {area_pixels} pixels")
                
            except Exception as e:
                print(f"Box {i}: SAM2 processing failed: {e}, using mock")
                # Fall back to mock processing
                area_pixels = int((x2 - x1) * (y2 - y1) * 0.7)
                area_nm2 = area_pixels / (scale_ratio ** 2) if scale_ratio else None
                padding = 10
                contour_points = [[
                    [x1 + padding, y1 + padding], 
                    [x2 - padding, y1 + padding], 
                    [x2 - padding, y2 - padding], 
                    [x1 + padding, y2 - padding]
                ]]
        else:
            # Use mock processing
            area_pixels = int((x2 - x1) * (y2 - y1) * 0.7)
            area_nm2 = area_pixels / (scale_ratio ** 2) if scale_ratio else None
            padding = 10
            contour_points = [[
                [x1 + padding, y1 + padding], 
                [x2 - padding, y1 + padding], 
                [x2 - padding, y2 - padding], 
                [x1 + padding, y2 - padding]
            ]]
            print(f"Box {i}: Mock segmentation - Area: {area_pixels} pixels")
        
        results.append({
            'box_id': i,
            'area_pixels': int(area_pixels),
            'area_nm2': float(area_nm2) if area_nm2 else None,
            'contours': contour_points,
            'box_coords': [x1, y1, x2, y2]
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
            # Load and process the image using OpenCV
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': 'Invalid image file'}), 400
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
