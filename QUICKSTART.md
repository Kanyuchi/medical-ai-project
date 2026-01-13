# Medical AI Project - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Upload to Google Drive
1. Zip the `medical-ai-project` folder on your Mac
2. Upload to your Google Drive
3. Extract in Drive (or upload the folder directly)

### Step 2: Download Datasets (ONE TIME ONLY)
1. Open Google Colab: https://colab.research.google.com/
2. Navigate to: File → Open notebook → Google Drive
3. Open: `medical-ai-project/setup_datasets.ipynb`
4. **IMPORTANT:** In Step 2 of the notebook, replace `YOUR_KAGGLE_USERNAME` with your actual Kaggle username
5. Run all cells sequentially
6. Wait 30-60 minutes for all datasets to download
7. ✓ Datasets will be stored permanently in your Drive!

### Step 3: Start Training
Choose one of the training notebooks:

**Option A - Chest X-Ray (Recommended to start)**
- Open: `chest-xray-classification/notebooks/chest_xray_classification.ipynb`
- Training time: 4-6 hours
- GPU needed: T4 or better

**Option B - Skin Lesion**
- Open: `skin-lesion-detection/notebooks/skin_lesion_detection.ipynb`
- Training time: 4-6 hours
- GPU needed: T4 or better

**Option C - Drug Discovery**
- Open: `drug-discovery/notebooks/drug_discovery.ipynb`
- Training time: 4-8 hours
- GPU needed: T4 or better

## 📊 Your Compute Resources
- Colab Pro+ Compute Units: **1,131.86 units**
- This is enough for **ALL THREE projects** with plenty to spare!

## ⚙️ Colab Settings
Before running any notebook:
1. Runtime → Change runtime type
2. Hardware accelerator: **GPU**
3. GPU type: T4, A100, or V100 (whichever is available)
4. Click Save

## 📁 Expected Dataset Sizes
- Chest X-Ray: ~3-5 GB
- Skin Lesion: ~5-8 GB
- Drug Discovery: ~1-2 GB
- **Total: ~10-15 GB** (make sure you have space in Google Drive)

## 🔑 Kaggle API Setup
Your API key is already in the setup notebook, but you need to add your username:

1. Go to: https://www.kaggle.com/settings/account
2. Find your username (it's at the top of the page)
3. In `setup_datasets.ipynb`, replace:
   ```python
   kaggle_username = "YOUR_KAGGLE_USERNAME"  # <- Replace this
   ```

## ⚠️ Important Notes

### Security
- Your API key is in the setup notebook
- Don't share this notebook publicly
- After datasets are downloaded, you can remove the API key from the notebook if desired

### Training Tips
1. **Start with Chest X-Ray** - It's the fastest and simplest
2. **One model at a time** - Don't run multiple notebooks simultaneously
3. **Enable GPU** - Always check Runtime → Change runtime type → GPU
4. **Monitor compute usage** - Check your remaining units in Colab
5. **Save checkpoints** - Models auto-save to Google Drive during training

### Common Issues

**"Kaggle API not found"**
- Make sure you ran the pip install cell in setup notebook

**"Dataset not found"**
- Verify your Kaggle username is correct
- Check that you've accepted the dataset terms on Kaggle website

**"Out of memory"**
- Reduce batch size in the notebook
- Restart runtime and try again

**"Runtime disconnected"**
- This happens with long training sessions
- Your work is saved in Google Drive
- Just reconnect and continue

## 📈 Training Progress Tracking

Each notebook will show:
- Loss curves (training & validation)
- Accuracy metrics
- Confusion matrices
- Sample predictions
- Model interpretability (GradCAM)

All visualizations are saved to your Google Drive automatically.

## 🎯 Next Steps After Training

1. **Evaluate your model** - Run the evaluation cells in each notebook
2. **Try different datasets** - Experiment with the alternative datasets
3. **Tune hyperparameters** - Adjust learning rate, batch size, etc.
4. **Compare architectures** - Try different model backbones
5. **Deploy** - Export models for inference

## 📚 Learning Resources

- Model interpretability: GradCAM shows what the model "sees"
- Class imbalance: Weighted loss and focal loss techniques
- Transfer learning: Fine-tuning pre-trained models
- Graph Neural Networks: Molecular property prediction

## 💬 Questions?

Check the main README.md for detailed documentation.

---

**Ready to start?** Open `setup_datasets.ipynb` in Google Colab!
