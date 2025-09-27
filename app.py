from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Simple test app for deployment
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mitochondria Segmentation - Test</title>
    </head>
    <body>
        <h1>Mitochondria Segmentation App</h1>
        <p>This is a test deployment to verify the app works.</p>
        <p>Status: ✅ App is running successfully!</p>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    return jsonify({'success': True, 'message': 'Upload endpoint working'})

@app.route('/process', methods=['POST'])
def process_boxes():
    return jsonify({'success': True, 'message': 'Process endpoint working'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
