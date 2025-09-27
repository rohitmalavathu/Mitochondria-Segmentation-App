# Mitochondria Segmentation Web Application

A web-based tool for analyzing mitochondrial structures using AI-powered segmentation with SAM2 (Segment Anything Model 2).

## Features

- **Image Upload**: Upload microscopy images in various formats
- **Multiple Box Drawing**: Draw multiple bounding boxes around regions of interest
- **AI Segmentation**: Automatic segmentation of mitochondria using fine-tuned SAM2 model
- **Area Calculation**: Calculate segmented area in both pixels and nanometers
- **Scale Reference**: Draw a 500nm reference line for accurate measurements
- **Real-time Results**: View segmentation results and area calculations instantly

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure SAM2 is properly set up**:
   The application uses the existing SAM2 model files in the `sam2/` directory.

## Usage

1. **Start the Web Application**:
   ```bash
   python app.py
   ```

2. **Open in Browser**:
   Navigate to `http://localhost:5000` in your web browser.

3. **Using the Tool**:
   - **Upload Image**: Click "Choose Image File" to upload a microscopy image
   - **Set Scale**: Switch to "Scale Line" mode and draw a 500nm reference line
   - **Draw Boxes**: Switch to "Draw Boxes" mode and draw bounding boxes around mitochondria
   - **Process**: Click "Process Segmentation" to analyze the selected regions
   - **View Results**: See area calculations and segmentation results

## Tool Modes

### Draw Boxes Mode
- Click and drag to create rectangular bounding boxes
- Each box will be processed for mitochondrial segmentation
- Boxes are numbered for easy identification

### Scale Line Mode
- Draw a line representing 500nm in your image
- The tool automatically calculates the pixel-to-nanometer ratio
- This enables accurate area measurements in real units

## API Endpoints

- `POST /upload`: Upload an image file
- `POST /process`: Process bounding boxes for segmentation
- `POST /calculate_scale`: Calculate scale ratio from reference line

## Technical Details

- **Backend**: Flask web framework
- **AI Model**: Fine-tuned SAM2 (Segment Anything Model 2)
- **Image Processing**: OpenCV and PIL
- **Frontend**: HTML5 Canvas with JavaScript
- **File Support**: Common image formats (PNG, JPG, TIFF, etc.)

## File Structure

```
├── app.py                 # Flask web application
├── templates/
│   └── index.html        # Web interface
├── uploads/              # Uploaded images (created automatically)
├── requirements.txt      # Python dependencies
└── sam2/                 # SAM2 model files
```

## Troubleshooting

1. **Model Loading Issues**: Ensure you have internet connection for downloading model weights
2. **Memory Issues**: Large images may require more RAM; consider resizing before upload
3. **Browser Compatibility**: Use modern browsers with HTML5 Canvas support

## Notes

- The application automatically resizes images to 512x512 for display
- Original image dimensions are preserved for accurate processing
- Scale calculations are based on the 500nm reference line
- Segmentation results include both pixel counts and area in square nanometers
