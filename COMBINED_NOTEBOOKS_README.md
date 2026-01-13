# COMBINED DATASETS - Training Notebooks

## 🎯 You Now Have COMBINED Dataset Notebooks!

### 📂 New Notebooks Created:

1. **chest_xray_COMBINED.ipynb**
   - Location: `chest-xray-classification/notebooks/`
   - Combines: COVID-19 Radiography (21,165) + Pneumonia (5,856)
   - **Total: ~27,000 images**
   - Classes: Normal, Pneumonia, COVID, Lung_Opacity
   - Training time: 8-10 hours

2. **skin_lesion_COMBINED.ipynb** 
   - Location: `skin-lesion-detection/notebooks/`
   - Combines: HAM10000 (10,015) + ISIC 2019 (2,351)
   - **Total: ~12,366 images**
   - Classes: 9 skin lesion types
   - Training time: 6-8 hours

## 🚀 How to Use:

1. **Upload to Google Colab**:
   - Go to https://colab.research.google.com/
   - File → Upload notebook
   - Select chest_xray_COMBINED.ipynb OR skin_lesion_COMBINED.ipynb

2. **Set GPU**:
   - Runtime → Change runtime type → GPU → Save

3. **Run All**:
   - Runtime → Run all
   - Authorize Google Drive when prompted

4. **Wait for Training**:
   - Chest X-Ray: 8-10 hours
   - Skin Lesion: 6-8 hours

## ✅ Features:

- ✅ Automatic data loading from both datasets
- ✅ Class mapping and label unification
- ✅ Class imbalance handling with weighted loss
- ✅ Strong data augmentation
- ✅ Progress bars for training
- ✅ Automatic model saving (best accuracy)
- ✅ Training curves visualization
- ✅ Confusion matrix
- ✅ Classification report
- ✅ Per-class accuracy
- ✅ Training summary JSON export

## 📊 Expected Results:

### Chest X-Ray:
- Expected accuracy: 90-95% (improved due to more data!)
- Best class: Normal (most samples)
- Challenging: COVID vs Viral Pneumonia

### Skin Lesion:
- Expected accuracy: 80-88%
- Best class: nevus (most samples)
- Challenging: Similar-looking lesions

## 🔧 Configuration:

Both notebooks use:
- Image size: 224x224
- Batch size: 32
- Learning rate: 0.0001
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau
- Architecture: ResNet50 (pretrained on ImageNet)

## 📝 Old Notebooks:

The original single-dataset notebooks are still there:
- `chest_xray_classification.ipynb` (COVID-19 only)
- `skin_lesion_detection.ipynb` (HAM10000 only)

These will be deleted to keep directories clean.

## 💡 Next Steps:

1. Start with chest_xray_COMBINED.ipynb (recommended)
2. Upload to Colab
3. Run all cells
4. Wait 8-10 hours
5. View amazing results!

Your compute: 1,131.86 units (plenty for both!)
