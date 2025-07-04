import os
import pandas as pd

def test_file_exists():
    for fname in [
        'models/synthetic_binary.csv',
        'models/synthetic_multiclass.csv',
        'models/synthetic_regression.csv',
    ]:
        assert os.path.exists(fname), f"File not found: {fname}"

def test_schema():
    df = pd.read_csv('models/synthetic_binary.csv')
    assert df.shape[1] == 11, "Expected 10 features + 1 target"
    for i in range(8):
        assert f'num_{i}' in df.columns
    for i in range(2):
        assert f'cat_{i}' in df.columns
    assert 'target' in df.columns

def test_types():
    df = pd.read_csv('models/synthetic_binary.csv')
    # Numeric columns should be convertible to float
    for i in range(8):
        pd.to_numeric(df[f'num_{i}'], errors='raise')
    # Categorical columns should be string
    for i in range(2):
        assert df[f'cat_{i}'].dtype == object
    # Target should be int for binary
    assert pd.api.types.is_integer_dtype(df['target']) 