# GitHub Codespace Setup for Mitochondria Segmentation App

## 🚀 Quick Start

1. **Start the Codespace** (if not already running)
2. **Run the startup script:**
   ```bash
   ./startup.sh
   ```
3. **Access your app:** https://humble-robot-4jw45465q9ggh5jrx-9669.app.github.dev/

## 🔄 If the App Goes Down

### Method 1: Restart Everything
```bash
pkill -f "python app.py"
./startup.sh
```

### Method 2: Manual Restart
```bash
pkill -f "python app.py"
nohup python app.py > app.log 2>&1 &
```

## 🌐 Sharing with Collaborators

1. **Make sure the app is running:**
   ```bash
   ps aux | grep "python app.py"
   ```

2. **Share the URL:** https://humble-robot-4jw45465q9ggh5jrx-9669.app.github.dev/

3. **Make port public:** In Codespace, go to "Ports" tab and click the globe icon next to port 9669

## ⚠️ Important Notes

- **Codespaces sleep after inactivity** - collaborators may see "page can't be found"
- **Solution:** Restart the Codespace and run `./startup.sh`
- **Keep-alive:** Run `./keepalive.sh` in background to prevent sleep
- **Check logs:** `tail -f app.log` to see what's happening

## 🛠️ Troubleshooting

### App won't start:
```bash
tail -20 app.log
```

### Port not accessible:
1. Check if port 9669 is running: `lsof -i :9669`
2. Make port public in Codespace Ports tab
3. Restart the app: `./startup.sh`

### Codespace sleeping:
1. Resume the Codespace from GitHub
2. Run `./startup.sh`
3. Share the new URL with collaborators
