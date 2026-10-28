# Insider Threat Detection Using Deep Learning

A deep-learning project for classifying network traffic using a CNN-LSTM model, with LIME-based explainability and an interactive Streamlit dashboard.

## Features

- Network traffic classification using CICIDS2017.
- Data cleaning and numerical feature preprocessing.
- CNN-LSTM model training using TensorFlow/Keras.
- Evaluation using accuracy, precision, recall, F1-score, and a confusion matrix.
- Local explanations for individual predictions using LIME.
- Interactive CSV upload and prediction dashboard.

## Reported Results

From the current training run:

| Metric | Result |
|---|---:|
| Test accuracy | 93.36% |
| Macro F1-score | 63.62% |
| Weighted F1-score | 93.33% |

Performance varies by class. The macro F1-score indicates that rare attack classes need particular attention. These results belong to this implementation and do not establish that the original thesis results have been reproduced.

## Project Structure

```text
insider-threat-detection/
├── app.py
├── requirements.txt
├── src/
│   ├── inspect_dataset.py
│   ├── train.py
│   └── explain.py
├── data/
│   └── MachineLearningCSV/
├── models/
└── outputs/
```

## Setup

Create and activate a virtual environment on Windows:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download the CICIDS2017 dataset and place the required CSV files in `data/MachineLearningCSV/`.

Inspect the dataset:

```powershell
python src/inspect_dataset.py
```

Train the model:

```powershell
python src/train.py
```

Generate a LIME explanation:

```powershell
python src/explain.py
```

Launch the dashboard:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Limitations

- Rare classes may have substantially lower detection performance.
- Uploaded-file predictions are not independent evaluation results unless ground-truth labels and a separate evaluation procedure are used.
- LIME provides local approximations of model behaviour, not proof of causation.
- This is an academic prototype, not a production-ready security system.

## Data and Model Files

The dataset, trained model, and generated outputs are not included in the Git repository. They must be prepared or generated locally before using the complete application.

## Author

Gufran Ahmed

GitHub: https://github.com/gufrxn999
