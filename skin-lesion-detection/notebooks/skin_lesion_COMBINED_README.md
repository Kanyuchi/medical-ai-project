# Skin Lesion COMBINED Dataset Notebook

## Status: Ready to Use

The combined skin lesion notebook merges:
- HAM10000: 10,015 images
- ISIC 2019: 2,351 images  
- **TOTAL: ~12,366 images**

## Classes (9 total):
1. actinic keratosis
2. basal cell carcinoma
3. dermatofibroma
4. melanoma
5. nevus
6. pigmented benign keratosis
7. seborrheic keratosis
8. squamous cell carcinoma
9. vascular lesion

## Training Time: 6-8 hours on T4 GPU

## How to Use:
Upload `skin_lesion_COMBINED.ipynb` to Google Colab and run all cells.

The notebook handles:
- Loading from both datasets
- Mapping HAM10000 codes to full class names
- Combining ISIC 2019 folder structure
- Class balancing with weighted loss
- Full training loop
- Evaluation and visualization
