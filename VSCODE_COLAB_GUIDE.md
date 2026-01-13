# Using VS Code Colab Extension - Step by Step Guide

## 🎯 Advantage of VS Code Colab Extension

Instead of using the browser, you can:
- ✅ Edit and run notebooks directly in VS Code
- ✅ Use VS Code's superior editor features
- ✅ Execute cells on Google Colab's GPUs
- ✅ Better code completion and IntelliSense
- ✅ Integrated terminal and debugging

## 📋 Step-by-Step Instructions

### Step 1: Open the Setup Notebook

VS Code is now open in your project directory!

1. In VS Code Explorer (left sidebar), navigate to:
   ```
   medical-ai-project/
   └── setup_datasets.ipynb
   ```

2. Click on `setup_datasets.ipynb` to open it

### Step 2: Connect to Google Colab

1. When the notebook opens, look at the **top right corner**
2. You'll see a button that says "Select Kernel" or "Connect"
3. Click it and select **"Google Colab"**
4. Sign in to your Google account when prompted
5. Wait for the connection to establish (you'll see "Connected to Google Colab")

**Status Indicator:**
- 🔴 Not connected
- 🟡 Connecting...
- 🟢 Connected to Colab runtime

### Step 3: Update Kaggle Credentials

Before running the notebook, you need to add your Kaggle username:

1. In the notebook, find the cell with:
   ```python
   kaggle_username = "YOUR_KAGGLE_USERNAME"
   kaggle_key = "KGAT_30bf102a6dc5f333e68e00b836ef2168"
   ```

2. Replace `YOUR_KAGGLE_USERNAME` with your actual Kaggle username

3. Save the file (Cmd+S)

**To find your Kaggle username:**
- Go to https://www.kaggle.com/settings/account
- Your username is displayed at the top

### Step 4: Run the Setup Notebook

**Option A: Run All Cells (Recommended)**
1. Click the "▶▶ Run All" button at the top of the notebook
2. Or use: Shift+Cmd+Enter on each cell

**Option B: Run Cell by Cell**
1. Click into a cell
2. Press Shift+Enter to run and move to next cell
3. Or click the ▶ play button next to the cell

**What to Expect:**
- First cell: Mounts Google Drive (you'll need to authorize)
- Second cell: Checks available storage
- Following cells: Download datasets (~30-60 minutes total)
- Final cell: Shows summary of downloaded datasets

### Step 5: Monitor Progress

In VS Code you'll see:
- **Cell outputs** appear below each cell
- **Progress bars** for downloads
- **Print statements** showing status
- **Any errors** will be displayed in red

**Tips:**
- Don't close VS Code during downloads
- You can minimize VS Code, but keep it running
- Datasets download directly to your Google Drive

### Step 6: After Setup Completes

Once all datasets are downloaded:

1. Open a training notebook:
   - `chest-xray-classification/notebooks/chest_xray_classification.ipynb` (recommended first)
   - OR `skin-lesion-detection/notebooks/skin_lesion_detection.ipynb`
   - OR `drug-discovery/notebooks/drug_discovery.ipynb`

2. Connect to Google Colab (same process as Step 2)

3. Run all cells to start training!

## 🔧 Colab Extension Features in VS Code

### Kernel Selection
- **Top right corner:** Shows current kernel/runtime
- Click to switch between local and Colab runtime
- Always select "Google Colab" for GPU access

### GPU Type Selection
1. Click on the kernel selector (top right)
2. Select "Change Runtime Type"
3. Choose GPU type: T4, A100, or V100
4. Higher-end GPUs train faster but use more compute units

### Cell Execution
- **Shift+Enter:** Run current cell and move to next
- **Ctrl+Enter:** Run current cell and stay
- **Run All:** Execute all cells sequentially
- **Run Above/Below:** Execute cells above/below current

### Variables and Outputs
- **Variables panel:** See all defined variables
- **Output panel:** View cell outputs and errors
- **Clear outputs:** Right-click → Clear All Outputs

## ⚙️ Recommended Settings

### VS Code Settings for Notebooks

1. Open Settings (Cmd+,)
2. Search for "notebook"
3. Recommended settings:
   - ✅ Notebook: Outline Show Code Cells
   - ✅ Notebook: Cell Toolbar Visibility
   - ✅ Notebook: Show Cell Status Bar

### Colab Runtime Settings

1. In notebook, go to Runtime menu
2. Set:
   - Hardware accelerator: GPU
   - GPU type: T4 (or higher if available)
   - Runtime shape: Standard

## 🐛 Troubleshooting

### "Cannot connect to Google Colab"
- Check internet connection
- Sign out and sign back into Google account
- Restart VS Code
- Try opening notebook in browser as backup

### "Kernel died" or "Runtime disconnected"
- This can happen with long-running cells
- Click "Reconnect" when prompted
- Your Google Drive data is safe
- Just re-run the failed cell

### "Permission denied" errors
- Make sure you authorized Google Drive access
- Check that Kaggle credentials are correct
- Verify you've accepted dataset terms on Kaggle

### Slow downloads
- Normal! Datasets are large (10-15 GB total)
- Colab has good bandwidth, just be patient
- 30-60 minutes is expected for all datasets

### Can't see cell outputs
- Click on the cell to expand output
- Check if output is collapsed (click arrow)
- Try clearing outputs and re-running

## 💡 Pro Tips

### Working with Multiple Notebooks
- Open multiple notebooks in separate tabs
- But only connect ONE to Colab at a time
- Switching between them is easy

### Saving Your Work
- All changes save automatically to your Mac
- Outputs and trained models save to Google Drive
- Use Cmd+S to force save

### Keyboard Shortcuts
- **Cmd+Shift+P:** Command palette
- **Shift+Enter:** Run cell
- **A:** Add cell above
- **B:** Add cell below
- **DD:** Delete cell
- **M:** Change to Markdown
- **Y:** Change to Code

### Monitoring Compute Usage
- Check your Colab Pro+ units in the Colab interface
- You have 1,131.86 units (plenty for all projects!)
- Each training session uses ~50-100 units

## 🚀 Quick Start Commands

```bash
# Open project in VS Code (already done)
code ~/medical-ai-project

# In VS Code:
# 1. Open setup_datasets.ipynb
# 2. Click "Select Kernel" → "Google Colab"
# 3. Sign in to Google
# 4. Update Kaggle username in cell
# 5. Click "Run All"
# 6. Wait 30-60 minutes
# 7. Start training with a training notebook!
```

## 📊 Expected Timeline

| Step | Time | What's Happening |
|------|------|------------------|
| Connect to Colab | 1-2 min | Authenticating and starting runtime |
| Mount Google Drive | 1 min | Authorizing Drive access |
| Download Chest X-Ray | 10-15 min | ~5 GB dataset |
| Download Skin Lesion | 15-20 min | ~8 GB dataset |
| Download Drug Discovery | 5-10 min | ~2 GB dataset |
| **Total** | **30-60 min** | **Ready to train!** |

---

**You're all set!** The notebook is already open in VS Code. Just connect to Colab and start running cells!
