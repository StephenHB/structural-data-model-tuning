from typing import Tuple, Optional
import numpy as np
import pandas as pd

def generate_numeric_features(n_samples: int, n_numeric: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate numeric features from a standard normal distribution."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(n_samples, n_numeric)

def generate_categorical_features(n_samples: int, n_categorical: int, categories=None, seed: Optional[int] = None) -> np.ndarray:
    """Generate categorical features uniformly from the given categories."""
    if categories is None:
        categories = ['A', 'B', 'C']
    if seed is not None:
        np.random.seed(seed + 1)
    return np.random.choice(categories, size=(n_samples, n_categorical))

def create_dataframe(X_numeric: np.ndarray, X_categorical: np.ndarray) -> pd.DataFrame:
    """Combine numeric and categorical features into a DataFrame."""
    n_numeric = X_numeric.shape[1]
    n_categorical = X_categorical.shape[1]
    columns = [f'num_{i}' for i in range(n_numeric)] + [f'cat_{i}' for i in range(n_categorical)]
    X = np.concatenate([X_numeric, X_categorical.astype(str)], axis=1)
    return pd.DataFrame(X, columns=columns)

def generate_binary_target(n_samples: int, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed + 2)
    return np.random.randint(0, 2, size=n_samples)

def generate_multiclass_target(n_samples: int, n_classes: int = 4, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed + 3)
    return np.random.randint(0, n_classes, size=n_samples)

def generate_regression_target(n_samples: int, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed + 4)
    return np.random.randn(n_samples)

def generate_synthetic_data(
    n_samples: int = 1000,
    n_numeric: int = 8,
    n_categorical: int = 2,
    task: str = 'binary',
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a synthetic tabular dataset for a given task.
    task: 'binary', 'multiclass', or 'regression'
    """
    X_numeric = generate_numeric_features(n_samples, n_numeric, seed)
    X_categorical = generate_categorical_features(n_samples, n_categorical, seed=seed)
    df = create_dataframe(X_numeric, X_categorical)
    if task == 'binary':
        y = generate_binary_target(n_samples, seed)
    elif task == 'multiclass':
        y = generate_multiclass_target(n_samples, n_classes=4, seed=seed)
    elif task == 'regression':
        y = generate_regression_target(n_samples, seed)
    else:
        raise ValueError(f"Unknown task: {task}")
    df['target'] = y
    return df

def save_synthetic_data(
    out_dir: str = 'data',
    n_samples: int = 1000,
    n_numeric: int = 8,
    n_categorical: int = 2,
    seed: int = 42
):
    """Generate and save synthetic datasets for all tasks."""
    for task in ['binary', 'multiclass', 'regression']:
        df = generate_synthetic_data(n_samples, n_numeric, n_categorical, task, seed)
        df.to_csv(f"{out_dir}/synthetic_{task}.csv", index=False)
    print(f"Synthetic data saved to {out_dir}/synthetic_[binary|multiclass|regression].csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic tabular data for TabPFN fine-tuning.")
    parser.add_argument('--out_dir', type=str, default='data', help='Output directory')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--n_numeric', type=int, default=8, help='Number of numeric features')
    parser.add_argument('--n_categorical', type=int, default=2, help='Number of categorical features')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    save_synthetic_data(
        out_dir=args.out_dir,
        n_samples=args.n_samples,
        n_numeric=args.n_numeric,
        n_categorical=args.n_categorical,
        seed=args.seed
    ) 