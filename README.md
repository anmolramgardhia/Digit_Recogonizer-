# MNIST Digit Recognizer

A Convolutional Neural Network (CNN) that classifies handwritten digits (0–9) with 99%+ accuracy, served via a FastAPI endpoint and a browser-based drawing app.

---

## Quick Start Commands

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python src/train.py
```
Downloads MNIST automatically, trains for 10 epochs (~5 min on GPU / ~20 min on CPU), prints per-epoch metrics, and saves the best model to `models/mnist_cnn.pt`.

### 3. Evaluate
```bash
python src/evaluate.py
```
Generates confusion matrix, per-class report, and a grid of misclassified samples. It also **saves the classification report** to `tests/classification_report.txt`.

### 4. Run the API + drawing app
*Note: If you encounter multiprocessing or socket bind errors on Windows, run without `--reload` or use a different port.*
```bash
python -m uvicorn src.api:app --port 8080
```
Open `http://localhost:8080` in your browser — draw a digit on the canvas and click Predict.

### 5. Run tests
```bash
pytest tests/ -v
```

---

## Project structure

```
digit-recognizer/
├── data/                       <- Auto-downloaded by torchvision
├── notebooks/
│   ├── 01_eda.ipynb            <- Dataset exploration & visualisation
│   └── 02_experiments.ipynb    <- Architecture experiments & comparison
├── src/
│   ├── dataset.py              <- DataLoader, transforms, preprocessing
│   ├── model.py                <- CNN nn.Module (base + BatchNorm variant)
│   ├── train.py                <- Training loop & hyperparameters
│   ├── evaluate.py             <- Confusion matrix, error analysis
│   └── api.py                  <- FastAPI prediction endpoint
├── static/
│   └── index.html              <- HTML5 canvas drawing app
├── models/
│   └── mnist_cnn.pt            <- Saved weights (auto-generated)
├── tests/
│   ├── test_model.py           <- Unit tests
│   └── classification_report.txt <- Generated report from evaluate.py
├── requirements.txt
├── digit_recognizer_prd.pdf
└── README.md
```

---

## Model performance

| Model          | Test Accuracy | Parameters |
|----------------|---------------|------------|
| DigitCNN       | 99.x%         | ~224,834   |
| DigitCNNBN     | 99.x%         | ~225,090   |

*Run train.py to populate with your results.*  
Target: >= 99.0% test accuracy.

---

## CNN architecture

```
Input (1×28×28)
  → Conv2d(32, 3×3) → ReLU → MaxPool(2×2)   → 32×13×13
  → Conv2d(64, 3×3) → ReLU → MaxPool(2×2)   → 64×5×5
  → Flatten → 1600
  → Dropout(0.5)
  → Linear(1600→128) → ReLU
  → Linear(128→10) → LogSoftmax
```

---

## Key learnings

- **Convolutional layers** extract hierarchical features: edges → curves → digit structures
- **Max pooling** provides spatial invariance — model is robust to small shifts in digit position
- **Dropout** prevents memorisation by randomly zeroing neurons during training
- **Training curves** reveal overfitting before it hurts test performance
- **Confusion matrix** shows which pairs of digits are hardest (4/9, 3/8)

---

## Tech stack

Python · PyTorch · torchvision · scikit-learn · FastAPI · Pillow · pytest
