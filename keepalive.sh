#!/bin/bash

# Keep-alive script to prevent Codespace from sleeping
echo "🔄 Starting keep-alive script..."

while true; do
    # Make a small request to keep the app active
    curl -s http://localhost:9669/ > /dev/null 2>&1
    
    # Also touch a file to show activity
    touch .keepalive
    
    # Wait 5 minutes before next ping
    sleep 300
done
