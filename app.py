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
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for large files

# Error handler for file too large
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 100MB. Please compress your image or use a smaller file.'}), 413

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
        
        # Set environment variable for better error reporting
        import os
        os.environ['HYDRA_FULL_ERROR'] = '1'
        
        # Test import first
        try:
            from sam2.modeling.backbones.hieradet import Hiera
            print("✓ Hiera class imported successfully")
        except Exception as import_error:
            print(f"✗ Error importing Hiera: {import_error}")
            return False
        
        FINE_TUNED_MODEL_WEIGHTS = hf_hub_download(repo_id="rohitmalavathu/SAM2FineTunedMito", filename="fine_tuned_sam2_2000.torch")
        sam2_checkpoint = hf_hub_download(repo_id="rohitmalavathu/SAM2FineTunedMito", filename="sam2_hiera_small.pt")
        model_cfg = "sam2_hiera_s.yaml"
        
        print(f"Model config: {model_cfg}")
        print(f"Checkpoint: {sam2_checkpoint}")
        
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS, map_location=torch.device('cpu'), weights_only=False))
        
        # Enable optimizations for faster inference
        sam2_model.eval()
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Compile model for faster inference (PyTorch 2.0+)
        try:
            sam2_model = torch.compile(sam2_model, mode="reduce-overhead")
            print("✓ Model compiled for faster inference")
        except Exception as compile_error:
            print(f"⚠ Model compilation not available: {compile_error}")
        
        model_loaded = True
        print("Model loaded successfully with optimizations!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

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
        print(f"Box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        print(f"Image shape: {image.shape}")
        
        # Validate coordinates
        if x1 >= x2 or y1 >= y2:
            print(f"Invalid box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            continue
            
        if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            print(f"Box coordinates out of bounds: x1={x1}, y1={y1}, x2={x2}, y2={y2}, image_shape={image.shape}")
            continue
        
        cropped_image = image[y1:y2, x1:x2]
        print(f"Cropped image shape: {cropped_image.shape}")
        
        if cropped_image.size == 0:
            print(f"Empty cropped image for box {i}")
            continue
            
        original_height, original_width = cropped_image.shape[:2]
        
        # Resize for model input with optimization
        try:
            # Use faster interpolation for resize
            cropped_image_resized = cv2.resize(cropped_image, (256, 256), interpolation=cv2.INTER_LINEAR)
            print(f"Resized image shape: {cropped_image_resized.shape}")
        except Exception as resize_error:
            print(f"Error resizing image: {resize_error}")
            print(f"Cropped image details: shape={cropped_image.shape}, dtype={cropped_image.dtype}")
            continue
        
        # Convert to 3-channel if grayscale
        if cropped_image_resized.ndim == 2:
            cropped_image_resized = np.stack([cropped_image_resized] * 3, axis=-1)
        
        # Process with SAM2 (optimized)
        with torch.no_grad():
            # Only set image if it's different from last one (optimization)
            if i == 0 or not hasattr(predictor, '_last_image') or not np.array_equal(predictor._last_image, cropped_image_resized):
                predictor.set_image(cropped_image_resized)
                predictor._last_image = cropped_image_resized.copy()
            
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
        
        # Find boundaries of SAM2 segmentation for display
        segmentation_255 = segmentation_resized * 255
        contours, _ = cv2.findContours(segmentation_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert SAM2 segmentation boundaries to list for JSON serialization
        # Adjust boundaries to be relative to the full image coordinates
        contour_points = []
        for contour in contours:
            if contour is not None and len(contour) > 0:
                # Adjust boundary points to be relative to the full image
                adjusted_contour = contour.reshape(-1, 2) + [x1, y1]
                contour_points.append(adjusted_contour.tolist())
        
        print(f"Box {i}: SAM2 found {len(contour_points)} segmentation boundaries")
        for j, contour in enumerate(contour_points):
            print(f"  SAM2 boundary {j}: {len(contour)} points")
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

# Store for chunked uploads
chunk_storage = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-chunk', methods=['POST'])
def upload_chunk():
    try:
        chunk = request.files['chunk']
        chunk_index = int(request.form['chunkIndex'])
        total_chunks = int(request.form['totalChunks'])
        file_name = request.form['fileName']
        file_id = request.form['fileId']
        
        if file_id not in chunk_storage:
            chunk_storage[file_id] = {
                'chunks': {},
                'total_chunks': total_chunks,
                'file_name': file_name
            }
        
        # Store chunk
        chunk_storage[file_id]['chunks'][chunk_index] = chunk.read()
        
        print(f"Received chunk {chunk_index + 1}/{total_chunks} for file {file_name}")
        
        return jsonify({'success': True, 'chunk': chunk_index + 1})
        
    except Exception as e:
        print(f"Error uploading chunk: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/assemble-chunks', methods=['POST'])
def assemble_chunks():
    try:
        data = request.get_json()
        file_id = data['fileId']
        file_name = data['fileName']
        total_chunks = data['totalChunks']
        
        if file_id not in chunk_storage:
            return jsonify({'error': 'File not found'}), 404
        
        stored_data = chunk_storage[file_id]
        
        # Check if all chunks are received
        if len(stored_data['chunks']) != total_chunks:
            return jsonify({'error': f'Missing chunks. Received {len(stored_data["chunks"])}/{total_chunks}'}), 400
        
        # Assemble file
        import uuid
        file_extension = os.path.splitext(file_name)[1]
        filename = f"{uuid.uuid4()}{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            for i in range(total_chunks):
                f.write(stored_data['chunks'][i])
        
        print(f"Assembled file: {filename} ({os.path.getsize(filepath)} bytes)")
        
        # Process the assembled file (same as regular upload)
        try:
            # Load and process image
            image = cv2.imread(filepath)
            if image is None:
                # Try with PIL for TIFF and other formats
                try:
                    pil_image = Image.open(filepath)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    return jsonify({'error': 'Invalid image file'}), 400
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Calculate scaling
            canvas_size = 1024
            original_height, original_width = image_rgb.shape[:2]
            scale_factor = min(canvas_size / original_width, canvas_size / original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            
            # Resize image
            display_image = cv2.resize(image_rgb, (new_width, new_height))
            
            # Create canvas
            canvas_image = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
            start_x = (canvas_size - new_width) // 2
            start_y = (canvas_size - new_height) // 2
            canvas_image[start_y:start_y + new_height, start_x:start_x + new_width] = display_image
            display_image = canvas_image
            
            # Encode image
            pil_image = Image.fromarray(display_image)
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            # Clean up
            del chunk_storage[file_id]
            
            return jsonify({
                'success': True,
                'image': img_str,
                'original_shape': image.shape,
                'display_shape': display_image.shape,
                'filename': filename,
                'scale_factor': scale_factor,
                'image_offset': {'x': start_x, 'y': start_y},
                'image_size': {'width': new_width, 'height': new_height}
            })
            
        except Exception as e:
            print(f"Error processing assembled file: {e}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Error assembling chunks: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Clean up old files before processing new upload
        cleanup_old_uploads()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate a unique filename to avoid conflicts
        import uuid
        file_extension = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and process image
        image = cv2.imread(filepath)
        if image is None:
            # Try with PIL for TIFF and other formats that OpenCV might not support
            try:
                pil_image = Image.open(filepath)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                return jsonify({'error': 'Invalid image file'}), 400
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculate scaling to fit canvas while maintaining aspect ratio
        canvas_size = 1024
        original_height, original_width = image_rgb.shape[:2]
        
        # Calculate scale factor to fit image in canvas
        scale_factor = min(canvas_size / original_width, canvas_size / original_height)
        
        # Calculate new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Resize image maintaining aspect ratio
        display_image = cv2.resize(image_rgb, (new_width, new_height))
        
        # Create a canvas-sized image with the resized image centered
        canvas_image = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        
        # Calculate position to center the image
        start_x = (canvas_size - new_width) // 2
        start_y = (canvas_size - new_height) // 2
        
        # Place the resized image on the canvas
        canvas_image[start_y:start_y + new_height, start_x:start_x + new_width] = display_image
        display_image = canvas_image
        
        # Encode image for frontend
        try:
            pil_image = Image.fromarray(display_image)
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            return jsonify({'error': 'Error encoding image for display'}), 500
        
        return jsonify({
            'success': True,
            'image': img_str,
            'original_shape': image.shape,
            'display_shape': display_image.shape,
            'filename': filename,
            'scale_factor': scale_factor,
            'image_offset': {'x': start_x, 'y': start_y},
            'image_size': {'width': new_width, 'height': new_height}
        })
    
    except Exception as e:
        print(f"ERROR in upload_file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/process', methods=['POST'])
def process_boxes():
    # Clean up old files before processing
    cleanup_old_uploads()
    
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        
        boxes = data.get('boxes', [])
        scale_ratio = data.get('scale_ratio', None)
        filename = data.get('filename')
        
        print(f"Processing {len(boxes)} boxes for file: {filename}")
        
        if not boxes:
            return jsonify({'error': 'No boxes provided'}), 400
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
    except Exception as e:
        print(f"Error parsing request data: {e}")
        return jsonify({'error': 'Invalid request data'}), 400
    
    # Load model if not already loaded
    if not load_model():
        return jsonify({'error': 'Failed to load model'}), 500
    
    # Load original image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'Image file not found'}), 404
    
    image = cv2.imread(filepath)
    if image is None:
        # Try with PIL as fallback
        try:
            pil_image = Image.open(filepath)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            return jsonify({'error': f'Could not read image: {str(e)}'}), 400
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Use boxes directly without any scaling
    original_height, original_width = image_rgb.shape[:2]
    
    # Get scaling information from the upload data
    scale_factor = data.get('scale_factor', 1.0)
    image_offset = data.get('image_offset', {'x': 0, 'y': 0})
    image_size = data.get('image_size', {'width': 512, 'height': 512})
    
    # Convert display coordinates to original image coordinates
    # The image is displayed at 512x512 but we need to scale back to original size
    scale_x = original_width / image_size['width']
    scale_y = original_height / image_size['height']
    
    scaled_boxes = []
    for box in boxes:
        # Handle both list and dict formats
        if isinstance(box, dict):
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        else:
            x1, y1, x2, y2 = box
        
        # Adjust coordinates to account for image offset
        x1_adjusted = x1 - image_offset['x']
        y1_adjusted = y1 - image_offset['y']
        x2_adjusted = x2 - image_offset['x']
        y2_adjusted = y2 - image_offset['y']
        
        print(f"Original box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        print(f"Adjusted coordinates: x1={x1_adjusted}, y1={y1_adjusted}, x2={x2_adjusted}, y2={y2_adjusted}")
        print(f"Scale factors: scale_x={scale_x}, scale_y={scale_y}")
        
        # Scale to original image coordinates
        scaled_box = [
            int(x1_adjusted * scale_x),
            int(y1_adjusted * scale_y),
            int(x2_adjusted * scale_x),
            int(y2_adjusted * scale_y)
        ]
        print(f"Scaled box coordinates: {scaled_box}")
        
        # Validate scaled coordinates (be more lenient)
        if scaled_box[0] >= scaled_box[2] or scaled_box[1] >= scaled_box[3]:
            print(f"Invalid scaled box coordinates: {scaled_box}")
            continue
            
        # Clamp coordinates to image bounds instead of rejecting
        scaled_box[0] = max(0, min(scaled_box[0], original_width - 1))
        scaled_box[1] = max(0, min(scaled_box[1], original_height - 1))
        scaled_box[2] = max(scaled_box[0] + 1, min(scaled_box[2], original_width))
        scaled_box[3] = max(scaled_box[1] + 1, min(scaled_box[3], original_height))
        
        print(f"Clamped scaled box coordinates: {scaled_box}")
            
        scaled_boxes.append(scaled_box)
    
    # Process segmentation
    print(f"Processing {len(scaled_boxes)} scaled boxes for segmentation")
    if len(scaled_boxes) == 0:
        print("No valid boxes to process!")
        return jsonify({'success': True, 'results': []})
    
    try:
        results = process_segmentation(image_rgb, scaled_boxes, scale_ratio)
        print(f"Segmentation returned {len(results)} results")
        
        # Clean up uploaded file after successful processing
        try:
            os.remove(filepath)
            print(f"Cleaned up uploaded file: {filename}")
        except Exception as cleanup_error:
            print(f"Warning: Could not delete uploaded file {filename}: {cleanup_error}")
        
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/calculate_scale', methods=['POST'])
def calculate_scale():
    data = request.get_json()
    line_length_pixels = data.get('line_length_pixels', 0)
    image_scaling = data.get('image_scaling')
    reference_length_nm = 500  # 500nm reference line
    
    if line_length_pixels <= 0:
        return jsonify({'error': 'Invalid line length'}), 400
    
    # If we have image scaling info, convert display coordinates to original image coordinates
    if image_scaling:
        scale_factor = image_scaling.get('scale_factor', 1.0)
        # Convert from display coordinates to original image coordinates
        actual_line_length = line_length_pixels / scale_factor
        print(f"Scale conversion: display={line_length_pixels}px -> original={actual_line_length}px (factor={scale_factor})")
    else:
        actual_line_length = line_length_pixels
        print(f"No image scaling info, using display length: {actual_line_length}px")
    
    # Calculate pixels per nm: actual_line_length / 500nm
    scale_ratio = actual_line_length / reference_length_nm  # pixels per nm
    
    print(f"Scale calculation: line_length={actual_line_length}px, reference={reference_length_nm}nm, pixels_per_nm={scale_ratio}")
    
    return jsonify({
        'success': True,
        'scale_ratio': scale_ratio,
        'pixels_per_nm': scale_ratio,
        'nm_per_pixel': 1 / scale_ratio
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9669))
    app.run(debug=False, host='0.0.0.0', port=port)
