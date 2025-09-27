# GitHub Codespaces Setup Guide

## 🚀 How to Use Your App in Codespaces

### Step 1: Open in Codespaces
1. **Go to your GitHub repository**: `https://github.com/rohitmalavathu/MitochondriaSegmentation`
2. **Click the green "Code" button**
3. **Select "Codespaces" tab**
4. **Click "Create codespace on main"**

### Step 2: Wait for Setup
- Codespaces will automatically:
  - Install Python 3.9
  - Install all dependencies from `requirements.txt`
  - Set up the development environment
- This takes about 2-3 minutes

### Step 3: Run Your App
Once Codespaces is ready, open the terminal and run:
```bash
python app.py
```

### Step 4: Access Your App
- Codespaces will automatically forward port 5000
- Click on the "Ports" tab in VS Code
- Click the "Open in Browser" icon next to port 5000
- Your app will open in a new tab!

### Step 5: Share with Your Person
- Copy the public URL from the "Ports" tab
- Share this URL with your person
- They can access your app from anywhere!

## 🔧 Troubleshooting

### If the app doesn't start:
```bash
# Check if all dependencies are installed
pip install -r requirements.txt

# Try running on a different port
PORT=5001 python app.py
```

### If SAM2 model fails to load:
- The model will download automatically on first use
- This may take a few minutes
- Check the terminal for download progress

## 📝 Notes

- **Free tier**: 120 hours/month
- **Automatic shutdown**: After 30 minutes of inactivity
- **Data persistence**: Your code is saved in GitHub
- **Easy sharing**: Just share the public URL

## 🎯 Benefits

- ✅ No deployment complexity
- ✅ Can handle large SAM2 model
- ✅ Free to use
- ✅ Easy to share
- ✅ Professional development environment
