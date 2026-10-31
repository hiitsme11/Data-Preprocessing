# Student Performance Data Preprocessing

## Week 1 – Python for Machine Learning & Data Preprocessing

This beginner-friendly project demonstrates how to inspect and preprocess a small student-performance dataset using Python, Pandas, NumPy and Scikit-learn.

**Dataset note:** `student_performance_raw.csv` is a synthetic sample dataset created for learning. It is not real student data.

## What the project does
1. Loads a CSV dataset using Pandas.
2. Inspects rows, columns, data types and missing values.
3. Detects and removes duplicate records.
4. Fills missing numeric values using the median.
5. Fills missing categorical values using the mode when needed.
6. Selects relevant features for a simple student-result dataset.
7. Encodes categorical features using one-hot encoding.
8. Converts the Pass/Fail target into numeric labels.
9. Normalizes numeric input features using Min-Max scaling.
10. Saves a cleaned dataset and an EDA bar chart.

## Requirements
- Python 3.9+
- pandas
- numpy
- matplotlib
- scikit-learn

Install:
```bash
pip install -r requirements.txt
```

## Run
Keep `preprocess.py` and `student_performance_raw.csv` in the same folder, then run:
```bash
python preprocess.py
```

The processed CSV and chart are saved inside the `output` folder.

## Folder structure
```text
Yuvaintern_Week1_Student_Performance/
├── student_performance_raw.csv
├── preprocess.py
├── requirements.txt
├── README.md
└── output/
    ├── student_performance_cleaned.csv
    └── result_distribution.png
```

## Important learning note
The goal is to demonstrate preprocessing steps, not to claim that a predictive model has been trained. Model training is covered in Week 2.
