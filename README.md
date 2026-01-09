# Overhead Assessment
Sigmoid and tanh implementations with error detection using Maclaurin series.

## Compile & Run
```bash
gcc -o activation Overhead_Assessment/C/main.c -lm
./activation
```

## Functions
- `sigmoid(x)` - Sigmoid activation
- `tanh(x)` - Tanh activation
- `sigmoid_error_detection(x, &err)` - With fault detection
- `tanh_error_detection(x, &err)` - With fault detection



## Notes
- Input range: -10 to 10 (auto-clipped)
- 30 Maclaurin terms
- Error tolerance: 0.000001

# Training
# NASA Jet Engine RUL Prediction Enabled with Error Detection

CNN model to predict Remaining Useful Life of aircraft jet engines using NASA C-MAPSS dataset.

## Quick Start

### 1. Setup Environment
```bash
cd Training
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Data
Download NASA C-MAPSS dataset and place in `archive/CMaps/`:
- `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
- `train_FD002.txt`, `test_FD002.txt`, `RUL_FD002.txt`
- `train_FD003.txt`, `test_FD003.txt`, `RUL_FD003.txt`
- `train_FD004.txt`, `test_FD004.txt`, `RUL_FD004.txt`

### 3. Run
```bash
python training.py
```

## Project Structure
```
nasa-rul-prediction/
├── README.md
├── requirements.txt
├── training.py
├── models/              # Output
└── archive/CMaps/       # Dataset folder
```

## Requirements
- Python 3.12.3
- See `requirements.txt`


## Results
Models saved to `models/` folder with metrics (Accuracy, F1-score, Recall) and average inference time.
