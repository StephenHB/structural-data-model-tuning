# Synthetic Data Generation for TabPFN Fine-Tuning

This document describes the process for generating synthetic tabular data for use in the finetuning pipeline, as required by the PRD.

## 1. Data Schema
- **Features:** 8 numeric, 2 categorical
- **Target:** Binary, multiclass, or regression
- **Types:**
  - Numeric: float (standard normal distribution)
  - Categorical: string (A/B/C, uniform)
  - Target: int (0/1 for binary, 0-3 for multiclass), float for regression

## 2. Data Generation Process
- Use Python (numpy, pandas) to generate features and targets
- Numeric features: `np.random.randn(n_samples, n_numeric)`
- Categorical features: `np.random.choice(['A', 'B', 'C'], size=(n_samples, n_categorical))`
- Targets:
  - Binary: `np.random.randint(0, 2, size=n_samples)`
  - Multiclass: `np.random.randint(0, 4, size=n_samples)`
  - Regression: `np.random.randn(n_samples)`

## 3. Saving Data
- Save as CSV files in the `models/` directory:
  - `synthetic_binary.csv`
  - `synthetic_multiclass.csv`
  - `synthetic_regression.csv`

## 4. Reproducibility
- Set random seeds as needed for reproducibility.
- Document all parameters and code in the provided notebook.

## 5. Usage
- The generated data is ready for use in the TabPFN finetuning pipeline.
- See `notebooks/synthetic_data_generation.ipynb` for code and examples. 