# Medical AI Project

A comprehensive collection of healthcare AI applications using deep learning for medical imaging and drug discovery.

## Project Overview

This project implements three major medical AI applications:

1. **Chest X-Ray Classification** - COVID-19 and Pneumonia Detection
2. **Skin Lesion Detection** - Melanoma and Skin Cancer Classification
3. **Drug Discovery** - Molecular Property Prediction using Graph Neural Networks

**Estimated GPU Time:** 12-20 hours total (optimized for Google Colab Pro+)

## Features

- Medical imaging classification with class imbalance handling
- Model interpretability using GradCAM
- Graph Neural Networks for molecular analysis
- Pre-trained models with transfer learning
- Comprehensive evaluation metrics for medical applications
- Colab-optimized notebooks for efficient GPU usage

## Project Structure

```
medical-ai-project/
├── chest-xray-classification/
│   ├── data/              # Chest X-ray datasets
│   ├── models/            # Trained models
│   ├── notebooks/         # Jupyter notebooks
│   └── train.py           # Training script
├── skin-lesion-detection/
│   ├── data/              # Dermoscopic images
│   ├── models/            # Trained models
│   ├── notebooks/         # Jupyter notebooks
│   └── train.py           # Training script
├── drug-discovery/
│   ├── data/              # Molecular datasets
│   ├── models/            # Trained GNN models
│   └── notebooks/         # Jupyter notebooks
├── utils/
│   ├── data_utils.py      # Data loading utilities
│   ├── visualization.py   # Plotting functions
│   └── metrics.py         # Evaluation metrics
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Getting Started

### Prerequisites

- Google Colab Pro+ account (recommended for GPU access)
- Python 3.8+
- CUDA-compatible GPU (Google Colab provides this)

### Installation

1. Clone or download this project to your Google Drive

2. Open the desired notebook in Google Colab:
   - Chest X-Ray: `chest-xray-classification/notebooks/chest_xray_classification.ipynb`
   - Skin Lesion: `skin-lesion-detection/notebooks/skin_lesion_detection.ipynb`
   - Drug Discovery: `drug-discovery/notebooks/drug_discovery.ipynb`

3. Mount your Google Drive when prompted

4. Install dependencies (included in each notebook):
```python
!pip install -r requirements.txt
```

## Datasets

### 1. Chest X-Ray Classification

**Recommended Datasets:**
- COVID-19 Radiography Database: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- Chest X-Ray Images (Pneumonia): https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

**Classes:** Normal, COVID-19, Pneumonia

### 2. Skin Lesion Detection

**Recommended Datasets:**
- HAM10000: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- ISIC 2019: https://challenge.isic-archive.com/data/

**Classes:** Melanoma, Melanocytic nevus, Basal cell carcinoma, Actinic keratosis, Benign keratosis, Dermatofibroma, Vascular lesion

### 3. Drug Discovery

**Recommended Datasets:**
- QM9: Quantum mechanical properties
- ESOL: Aqueous solubility
- FreeSolv: Hydration free energy
- Lipophilicity: Octanol/water distribution
- BACE: Binding affinity

Available through DeepChem: `dc.molnet.load_esol()`, `dc.molnet.load_bace_classification()`

## Key Techniques

### Medical Imaging

- **Transfer Learning**: Using pre-trained ResNet50/EfficientNet models
- **Data Augmentation**: Rotation, flipping, color jittering for robustness
- **Class Imbalance Handling**:
  - Weighted loss functions
  - Focal loss
  - Stratified sampling
- **Interpretability**: GradCAM visualization to understand model decisions

### Drug Discovery

- **Graph Neural Networks**: GAT (Graph Attention Networks) for molecular representation
- **Molecular Featurization**: Converting SMILES to graph structures
- **Property Prediction**: Regression for continuous properties (solubility, binding affinity)

## Model Architectures

### Chest X-Ray & Skin Lesion
- **Base:** ResNet50 / EfficientNet-B4 (pre-trained on ImageNet)
- **Fine-tuning:** Last layers trainable
- **Output:** Softmax classifier

### Drug Discovery
- **Base:** Graph Attention Networks (GAT)
- **Layers:** 3-layer GNN with attention mechanism
- **Pooling:** Global mean + max pooling
- **Output:** Regression head for property prediction

## Training

### Google Colab Pro+ Setup

1. Go to Runtime → Change runtime type
2. Select GPU (T4, A100, or V100)
3. Run the notebook cells sequentially

### Estimated Training Times (on Colab GPU)

- Chest X-Ray Classification: 4-6 hours
- Skin Lesion Detection: 4-6 hours
- Drug Discovery: 4-8 hours

## Evaluation Metrics

### Classification Tasks
- Accuracy
- Precision, Recall, F1-Score (per-class)
- ROC-AUC
- Confusion Matrix
- Sensitivity & Specificity (medical focus)

### Regression Tasks (Drug Discovery)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Key Learning Outcomes

1. **Medical Imaging Fundamentals**
   - Handling medical image datasets
   - Understanding class imbalance in healthcare
   - Model interpretability for clinical applications

2. **Advanced Deep Learning**
   - Transfer learning with pre-trained models
   - Fine-tuning strategies
   - Graph Neural Networks

3. **Healthcare AI Ethics**
   - Importance of interpretability in medical AI
   - Understanding model biases
   - Evaluation metrics for healthcare applications

## Results Visualization

Each notebook includes:
- Training/validation curves
- Confusion matrices
- ROC curves
- GradCAM heatmaps (interpretability)
- Sample predictions with confidence scores

## Future Enhancements

- [ ] Multi-modal learning (combining different imaging modalities)
- [ ] Federated learning for privacy-preserving medical AI
- [ ] Real-time inference deployment
- [ ] Integration with DICOM medical imaging standards
- [ ] Uncertainty quantification for predictions

## References

### Papers
- ResNet: "Deep Residual Learning for Image Recognition" (He et al., 2016)
- EfficientNet: "EfficientNet: Rethinking Model Scaling for CNNs" (Tan & Le, 2019)
- GradCAM: "Grad-CAM: Visual Explanations from Deep Networks" (Selvaraju et al., 2017)
- GAT: "Graph Attention Networks" (Veličković et al., 2018)

### Datasets
- COVID-19 Radiography Database
- HAM10000 Skin Lesion Dataset
- ISIC Archive
- MoleculeNet Datasets

## License

This project is for educational purposes. Please cite appropriate datasets and papers when using this code.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional medical imaging modalities (MRI, CT scans)
- More drug discovery tasks
- Advanced interpretability techniques
- Deployment pipelines

## Contact

For questions or collaborations, please open an issue in the repository.

## Acknowledgments

- Google Colab for providing free GPU resources
- PyTorch and PyTorch Geometric teams
- Medical imaging dataset contributors
- RDKit for molecular informatics tools
