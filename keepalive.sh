#!/bin/bash

# Keep-alive script to prevent Codespace from sleeping
echo "🔄 Starting keep-alive script for Mitochondria Segmentation App..."

# Create log file
LOG_FILE="keepalive.log"
echo "$(date): Keep-alive script started" >> $LOG_FILE

# Function to check if Flask app is running
check_app() {
    if ps aux | grep -v grep | grep "python app.py" > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to restart Flask app
restart_app() {
    echo "$(date): Flask app not running, restarting..." >> $LOG_FILE
    pkill -f "python app.py" 2>/dev/null
    sleep 2
    nohup python app.py > app.log 2>&1 &
    sleep 3
    if check_app; then
        echo "$(date): Flask app restarted successfully" >> $LOG_FILE
    else
        echo "$(date): Failed to restart Flask app" >> $LOG_FILE
    fi
}

# Main loop
while true; do
    # Check if Flask app is running
    if ! check_app; then
        echo "$(date): Flask app not running, restarting..." >> $LOG_FILE
        restart_app
    fi
    
    # Make a small request to keep the app active
    if curl -s http://localhost:9669/ > /dev/null 2>&1; then
        echo "$(date): App is responding" >> $LOG_FILE
    else
        echo "$(date): App not responding, restarting..." >> $LOG_FILE
        restart_app
    fi
    
    # Touch a file to show activity
    touch .keepalive
    
    # Wait 2 minutes before next check (more frequent)
    sleep 120
done
