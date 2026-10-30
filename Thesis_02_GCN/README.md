# Flow GCN classifier

This project trains a Graph Convolutional Network (GCN) to classify network
flows as benign/background (class `0`) or malicious (class `1`). It trains on
`unified_captures_train.csv` and reports test accuracy, malicious-class F1,
per-class precision/recall/F1/support, and a confusion matrix for
`unified_captures_test.csv`.

## Requirements

- Python 3.10 or newer
- The Python packages listed in `requirements.txt`
- The training and test CSV files (included in this project)

The model uses `Dur`, `TotPkts`, `TotBytes`, `SrcBytes`, and any `State_*`
columns as node features. Flows with the same `SrcAddr` and `DstAddr` are
connected as neighbors. Labels are read from `BinaryLabel` if present;
otherwise `NormLabel` is used. CSV files must include the required features,
`SrcAddr`, `DstAddr`, and a supported label column.

## Install dependencies

In PowerShell, open this project folder and create a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, run the commands using the
virtual environment's Python directly:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements file installs NumPy, pandas, scikit-learn, PyTorch, and
PyTorch Geometric. Install Python first if it is not already available; the
official downloads are at [python.org/downloads](https://www.python.org/downloads/).

## Run training and evaluation

From the project folder, run:

```powershell
python train.py
```

By default, training runs for 10 epochs with a batch size of 256 and a hidden
layer size of 32. The script uses a CUDA GPU when PyTorch detects one;
otherwise, it runs on the CPU. The evaluation output includes a detailed
classification report for benign/background and malicious flows, plus a
confusion matrix whose rows are actual classes and columns are predicted
classes. The current training script does not save a model checkpoint.

You can change the training options from the command line:

```powershell
python train.py --epochs 20 --batch-size 128 --hidden-dim 64
```

To specify different CSV paths:

```powershell
python train.py --train-csv .\path\to\train.csv --test-csv .\path\to\test.csv
```

Run `python train.py --help` to see all available options.
