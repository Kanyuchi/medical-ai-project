# 🚀 Start Training - Quick Guide

## ✅ Setup Complete!

All datasets are downloaded and ready in your Google Drive:
- ✅ Chest X-Ray: 21,165 images
- ✅ Skin Lesion: 10,015 images
- ✅ Drug Discovery: 1,128 molecules

## 🎯 Ready to Train!

### Option 1: Chest X-Ray Classification (RECOMMENDED START)

**Why start here?**
- Fastest to train (4-6 hours)
- 4 classes: COVID, Lung Opacity, Normal, Viral Pneumonia
- 21,165 images
- Best for learning the workflow

**How to start:**

1. **Open Google Colab**: https://colab.research.google.com/

2. **Upload the notebook**:
   - File → Upload notebook
   - Navigate to: `medical-ai-project/chest-xray-classification/notebooks/`
   - Select: `chest_xray_classification.ipynb`

3. **Set GPU runtime**:
   - Runtime → Change runtime type
   - Hardware accelerator: **GPU**
   - GPU type: T4 or better
   - Click Save

4. **Run all cells**:
   - Runtime → Run all
   - OR press Shift+Enter through each cell

5. **What will happen**:
   - Cell 1: Check GPU ✅
   - Cell 2: Mount Google Drive (authorize when prompted)
   - Cell 3-8: Install dependencies and setup
   - Cell 9: **Load 21,165 images** (this will take 2-3 minutes)
   - Cell 10-13: Build and configure model
   - Cell 14-16: **TRAINING STARTS** (4-6 hours)
   - Cell 17-19: View results, plots, confusion matrix

6. **Monitor progress**:
   - Each epoch shows: Train Loss, Train Acc, Val Loss, Val Acc
   - Best model saves automatically to Google Drive
   - 25 epochs total (~10-15 minutes per epoch)

### Option 2: Skin Lesion Detection

**7 classes of skin lesions**
- Training time: 4-6 hours
- 10,015 dermoscopic images
- More complex due to class imbalance

### Option 3: Drug Discovery

**Molecular property prediction**
- Training time: 4-8 hours
- 1,128 molecules with SMILES strings
- Uses Graph Neural Networks

## 📊 Training Tips

### GPU Selection
- **T4**: Good for all projects, 16GB VRAM
- **A100**: Fastest, but uses more compute units
- **V100**: Fast, good balance

### Managing Compute Units
You have **1,131.86 compute units**:
- Chest X-Ray: ~50-80 units
- Skin Lesion: ~50-80 units
- Drug Discovery: ~60-100 units
- **Total needed: ~200-300 units** (you have plenty!)

### During Training

**Normal behavior:**
- First epoch is slowest (model initialization)
- Loss should decrease over time
- Accuracy should increase
- Validation metrics may fluctuate slightly

**What to watch:**
- ✅ Train loss decreasing
- ✅ Val accuracy improving
- ⚠️ If val loss increases while train loss decreases = overfitting
- ⚠️ If both losses stop improving = training plateaued

### If Training Disconnects

**Don't panic!** Your model is saved:
1. Reconnect to Colab
2. Re-run cells 1-13 to setup
3. Load the saved model:
```python
model.load_state_dict(torch.load('./models/best_chest_xray_model.pth'))
```
4. Continue training or evaluate

## 📈 Expected Results

### Chest X-Ray
- **Expected accuracy**: 85-95%
- **Training time**: 4-6 hours
- **Best practices**:
  - Class imbalance handled with weighted loss ✅
  - Data augmentation enabled ✅
  - Transfer learning from ImageNet ✅

### Skin Lesion
- **Expected accuracy**: 75-85%
- **Training time**: 4-6 hours
- **Challenges**:
  - Severe class imbalance
  - Visual similarity between classes
  - Requires more epochs or data augmentation

### Drug Discovery
- **Expected MAE**: 0.5-1.0
- **Expected R²**: 0.7-0.9
- **Training time**: 4-8 hours
- **Note**: Regression task (not classification)

## 🎉 After Training

Once training completes, you'll have:

1. **Trained model** saved in Google Drive
2. **Training curves** showing loss/accuracy over time
3. **Confusion matrix** showing per-class performance
4. **Classification report** with precision/recall/F1
5. **Saved plots** in your project folder

## 🔄 Training Multiple Models

**Sequential approach (recommended):**
1. Train Chest X-Ray first (4-6 hours)
2. Close that notebook
3. Train Skin Lesion (4-6 hours)
4. Close that notebook
5. Train Drug Discovery (4-8 hours)

**Why sequential?**
- Only uses one GPU at a time
- Easier to monitor
- Less chance of disconnection
- Total time: ~12-20 hours over 2-3 days

## ⚡ Quick Start Command

**Copy-paste this into your first notebook cell after mounting Drive:**

```python
# Quick verification that data is loaded correctly
from pathlib import Path
import numpy as np

data_path = Path('./data/COVID-19_Radiography_Dataset')
classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

for c in classes:
    imgs = list((data_path / c / 'images').glob('*.png'))
    print(f"{c}: {len(imgs)} images")
```

Expected output:
```
COVID: 3616 images
Lung_Opacity: 6012 images
Normal: 10192 images
Viral Pneumonia: 1345 images
```

If you see this, you're ready to train! 🚀

## 📞 Troubleshooting

**"No such file or directory"**
- Make sure Google Drive is mounted
- Check that setup notebook completed successfully
- Verify project path is correct

**"CUDA out of memory"**
- Reduce BATCH_SIZE from 32 to 16
- Restart runtime
- Close other Colab notebooks

**"Runtime disconnected"**
- Normal for long training sessions
- Your progress is saved
- Just reconnect and continue

**Training too slow**
- Check you selected GPU (not CPU)
- Try a different GPU type (A100 is fastest)
- Reduce image size or batch size

---

## 🎯 YOU'RE READY!

**Next step**: Open `chest_xray_classification.ipynb` in Google Colab and click "Run All"

**Estimated time to first results**: 4-6 hours

**Your compute budget**: 1,131.86 units (more than enough!)

Good luck with your medical AI training! 🏥🤖
