# 🚀 Deployment Instructions for Google Cloud Run

## Step 1: Create GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Repository name: `mitochondria-segmentation-app`
5. Description: `Web app for mitochondria segmentation using SAM2 with measurement tools`
6. Make it **Public** (required for free Cloud Run deployment)
7. **DO NOT** initialize with README, .gitignore, or license (we already have these)
8. Click "Create repository"

## Step 2: Push Code to GitHub

Run these commands in your terminal:

```bash
# Add the GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/mitochondria-segmentation-app.git

# Push the code to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Set Up Google Cloud

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Note your **Project ID** (you'll need this for deployment)

## Step 4: Install Google Cloud CLI

### On macOS:
```bash
# Using Homebrew
brew install google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

### On Windows:
Download and install from: https://cloud.google.com/sdk/docs/install

### On Linux:
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

## Step 5: Authenticate and Configure

```bash
# Login to Google Cloud
gcloud auth login

# Set your project (replace YOUR_PROJECT_ID with your actual project ID)
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## Step 6: Deploy to Cloud Run

### Option A: Using the deployment script (Recommended)

1. Edit the `deploy.sh` file and replace `your-project-id` with your actual Google Cloud project ID
2. Run the deployment script:
```bash
./deploy.sh
```

### Option B: Manual deployment

```bash
# Replace YOUR_PROJECT_ID with your actual project ID
gcloud run deploy mitochondria-segmentation \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --max-instances 10 \
  --port 8080
```

## Step 7: Access Your App

After deployment, you'll get a URL like:
`https://mitochondria-segmentation-xxxxx-uc.a.run.app`

Visit this URL to test your application!

## Important Notes

### SAM2 Model Files
Make sure the SAM2 model files are included in your repository. If they're too large for GitHub:

1. **Option 1**: Use Git LFS (Large File Storage)
```bash
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Add LFS tracking for model files"
```

2. **Option 2**: Download models during deployment
Add this to your Dockerfile:
```dockerfile
# Download SAM2 models during build
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam2-hiera_large', 'sam2_hiera_large.pt', local_dir='sam2')"
```

### Environment Variables
If you need to set environment variables:
```bash
gcloud run services update mitochondria-segmentation \
  --region us-central1 \
  --set-env-vars "VARIABLE_NAME=value"
```

### Monitoring
- View logs: `gcloud run logs tail mitochondria-segmentation --region us-central1`
- View service details: `gcloud run services describe mitochondria-segmentation --region us-central1`

## Troubleshooting

### Common Issues:

1. **Out of memory**: Increase memory allocation in deployment command
2. **Timeout**: Increase timeout value
3. **Model not found**: Ensure SAM2 files are properly included
4. **Build fails**: Check that all dependencies are in requirements.txt

### Cost Optimization:
- Cloud Run only charges when requests are being processed
- Free tier includes 2 million requests per month
- Consider setting max instances to limit costs

## Next Steps

1. Test all functionality on the deployed app
2. Set up custom domain (optional)
3. Configure monitoring and alerts
4. Set up CI/CD for automatic deployments

## Support

If you encounter issues:
1. Check the Cloud Run logs
2. Verify all files are properly committed to GitHub
3. Ensure your Google Cloud project has billing enabled
4. Check that all required APIs are enabled
