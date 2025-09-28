#!/bin/bash

# Startup script for Mitochondria Segmentation App
echo "Starting Mitochondria Segmentation App..."

# Kill any existing Python processes
pkill -f "python app.py" 2>/dev/null

# Wait a moment
sleep 2

# Start the Flask app in the background
nohup python app.py > app.log 2>&1 &

# Get the process ID
PID=$!

# Wait a moment for the app to start
sleep 3

# Check if the app is running
if ps -p $PID > /dev/null; then
    echo "✅ Flask app started successfully (PID: $PID)"
    echo "📱 App is running on port 9669"
    echo "🌐 Access your app at: https://humble-robot-4jw45465q9ggh5jrx-9669.app.github.dev/"
else
    echo "❌ Failed to start Flask app"
    echo "📋 Check app.log for error details:"
    tail -20 app.log
fi
