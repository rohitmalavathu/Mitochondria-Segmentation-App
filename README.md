# Mitochondria Segmentation Web App

A web application for analyzing mitochondria in microscopy images using SAM2 (Segment Anything Model 2) with measurement tools for PSD length and vesicle counting.

## Features

- **Image Upload**: Upload microscopy images (TIF, PNG, JPG)
- **Scale Calibration**: Draw a 500nm reference line for accurate measurements
- **Mitochondria Segmentation**: Draw bounding boxes around mitochondria for AI segmentation
- **PSD Length Measurement**: Measure Postsynaptic Density width
- **Vesicle Counting**: Count presynaptic vesicles organized by neurons
- **Real-time Results**: View segmentation contours, areas, and measurements

## Tools

1. **Scale Line**: Draw a 500nm reference line for accurate measurements
2. **Draw Boxes**: Click and drag to create bounding boxes around mitochondria
3. **PSD Length**: Draw a line across a PSD region to measure its width
4. **Vesicle Count**: Click to place dots on presynaptic vesicles and organize by neurons

## Requirements

- Python 3.9+
- SAM2 model files
- OpenCV
- PyTorch
- Flask

## Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd MitochondriaSegmentation-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure SAM2 model files are in the `sam2/` directory

4. Run the application:
```bash
python app.py
```

5. Open your browser to `http://localhost:5000`

## Deployment

### Google Cloud Run

1. Install Google Cloud CLI
2. Authenticate: `gcloud auth login`
3. Set project: `gcloud config set project YOUR_PROJECT_ID`
4. Build and deploy:
```bash
gcloud run deploy mitochondria-segmentation \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600
```

## Usage

1. **Upload Image**: Click "Choose File" and select a microscopy image
2. **Set Scale**: Use Scale Line tool to draw a 500nm reference line
3. **Draw Boxes**: Use Draw Boxes tool to create bounding boxes around mitochondria
4. **Process**: Click "Process Segmentation" to run AI segmentation
5. **Measure PSD**: Use PSD Length tool to measure postsynaptic density width
6. **Count Vesicles**: Use Vesicle Count tool to count and organize vesicles by neurons

## Technical Details

- **Backend**: Flask with SAM2 model integration
- **Frontend**: HTML5 Canvas with JavaScript
- **Image Processing**: OpenCV for contour detection and image manipulation
- **AI Model**: SAM2 for mitochondria segmentation
- **Deployment**: Docker container on Google Cloud Run

## File Structure

```
├── app.py                 # Flask backend
├── templates/
│   └── index.html        # Frontend interface
├── sam2/                 # SAM2 model files
├── uploads/              # Uploaded images (temporary)
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
└── README.md            # This file
```

## License

This project is for research and educational purposes.