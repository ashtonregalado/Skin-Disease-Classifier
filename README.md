# Skin Disease Classifier

A deep learning project that classifies 22 skin disease categories from an uploaded image using a PyTorch transfer learning pipeline and a Streamlit web app.

by Prince Cedrik C. Herrera & Sean Ashton V. Regalado

## What the project does

- Trains a MobileNetV2-based image classifier on the `SkinDisease/train/` dataset.
- Evaluates the trained model on `SkinDisease/test/`.
- Provides a Streamlit UI where users can upload a skin image and see the top predictions with confidence scores.

## Project structure

```text
Skin-Disease-Classifier/
├── app.py                 # Streamlit UI for inference
├── dataset.py             # Data loading, preprocessing, augmentation
├── evaluate.py            # Test-set evaluation and confusion matrix
├── model.py               # MobileNetV2 backbone and classifier head
├── train.py               # Training loop with Phase A and Phase B
├── models/
│   ├── best_model_phaseA.pth
│   ├── best_model_final.pth
│   ├── classes.json
│   ├── history.json
│   └── training_curves.png
├── SkinDisease/ (this is the dataset from kaggle)
│   ├── train/
│   └── test/
├── requirements.txt
└── README.md
```

## Supported Skin Conditions (22 classes)

Acne, Actinic Keratosis, Benign Tumors, Bullous, Candidiasis,
Drug Eruption, Eczema, Infestations/Bites, Lichen, Lupus, Moles,
Psoriasis, Rosacea, Seborrheic Keratoses, Skin Cancer,
Sun/Sunlight Damage, Tinea, Unknown/Normal, Vascular Tumors,
Vasculitis, Vitiligo, Warts

## How to run the app

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the trained model exists:

- `models/best_model_final.pth`
- `models/classes.json`

3. Launch the Streamlit app:

```bash
streamlit run app.py
```

4. Upload a skin image in the browser and click **Predict**.

## How to train a new model

If you want to retrain again:

```bash
python train.py
```

This runs two phases:

- **Phase A:** trains the classifier head while the backbone is frozen.
- **Phase B:** unfreezes part of the backbone and fine-tunes the network.

When training finishes, the script saves:

- `models/best_model_phaseA.pth`
- `models/best_model_final.pth`
- `models/history.json`
- `models/training_curves.png`

## How to evaluate the model

```bash
python evaluate.py
```

This prints:

- classification report
- overall accuracy
- per-class accuracy
- confusion matrix image saved in `models/confusion_matrix.png`

## Training details

- **Framework:** PyTorch
- **Backbone:** MobileNetV2 pretrained on ImageNet
- **Input size:** 224 x 224
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam
- **Scheduler:** ReduceLROnPlateau
- **UI:** Streamlit

## Notes

- CPU training can take several hours depending on your laptop.
- Keep `models/classes.json` if retraining model

## Requirements

- Python 3.8+
- torch
- torchvision
- streamlit
- scikit-learn
- matplotlib
- seaborn
- pillow
- numpy

## Dataset source

Kaggle Skin Disease Dataset: https://www.kaggle.com/datasets/pacificrm/skindiseasedataset
