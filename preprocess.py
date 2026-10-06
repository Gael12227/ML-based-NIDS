import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level',
]
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']
NUMERICAL_COLS = [c for c in COLUMNS[:-2] if c not in CATEGORICAL_COLS]  # 38 columns


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path} has {df.shape[1]} columns, expected {len(COLUMNS)}")
    df.columns = COLUMNS
    df = df.drop(columns=['difficulty_level'])
    # Strip whitespace only: keep original casing (e.g. flag 'SF') so the saved
    # pipeline matches raw data at inference time. Labels are compared lowercase.
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype(str).str.strip()
    df['attack_type'] = df['attack_type'].astype(str).str.strip().str.lower()
    return df


def make_encoder() -> OneHotEncoder:
    # sklearn >= 1.2 uses sparse_output; older versions use sparse
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


def save_matrix(arr: np.ndarray, names, path_stem: Path, fmt: str) -> None:
    if fmt == 'npy':
        np.save(path_stem.with_suffix('.npy'), arr)
    elif fmt == 'csv':
        pd.DataFrame(arr, columns=names).to_csv(path_stem.with_suffix('.csv'), index=False)
    elif fmt == 'parquet':
        pd.DataFrame(arr, columns=names).to_parquet(path_stem.with_suffix('.parquet'), index=False)


def convert_train_file(input_path: Path, output_path: Path) -> None:
    df = load_split(input_path)
    target = (df.pop('attack_type') != 'normal').astype(int).rename('target')

    pre = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', NUMERICAL_COLS),
            ('cat', make_encoder(), CATEGORICAL_COLS),
        ],
        verbose_feature_names_out=False,
    )
    features = pre.fit_transform(df)
    feature_names = list(pre.get_feature_names_out())
    result = pd.DataFrame(features, columns=feature_names)
    result[target.name] = target.to_numpy()

    if result.select_dtypes(exclude=np.number).shape[1] != 0:
        raise ValueError('Converted dataset still contains non-numeric columns')
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('Converted dataset contains NaN or infinite values')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f'Converted {input_path} -> {output_path}')
    print(f'Rows: {len(result)} | Features: {len(feature_names)} | target counts: {target.value_counts().to_dict()}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='datasets/KDDTest+.txt')
    ap.add_argument('--output', default='data2.csv')
    ap.add_argument('--raw-dir', default=None)
    ap.add_argument('--out-dir', default='data/processed')
    ap.add_argument('--format', choices=['npy', 'csv', 'parquet'], default='npy')
    args = ap.parse_args()

    if args.raw_dir is None:
        convert_train_file(Path(args.input), Path(args.output))
        return

    import joblib

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(raw_dir / 'KDDTrain+.txt')
    test_df = load_split(raw_dir / 'KDDTest+.txt')

    # Binary target: normal -> 0, everything else -> 1
    y_train = (train_df['attack_type'] != 'normal').astype(int).rename('label')
    y_test = (test_df['attack_type'] != 'normal').astype(int).rename('label')

    X_train_raw = train_df.drop(columns=['attack_type'])
    X_test_raw = test_df.drop(columns=['attack_type'])

    # Fit on TRAIN only; transform both. Numeric block first, then one-hot block.
    pre = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', make_encoder(), CATEGORICAL_COLS),
        ],
        verbose_feature_names_out=False,
    )
    X_train = pre.fit_transform(X_train_raw)
    X_test = pre.transform(X_test_raw)
    feature_names = list(pre.get_feature_names_out())

    # Informational: categories in test that never appeared in train (encoded as all zeros)
    ohe = pre.named_transformers_['cat']
    for col, known in zip(CATEGORICAL_COLS, ohe.categories_):
        unseen = sorted(set(X_test_raw[col]) - set(known))
        if unseen:
            print(f"[info] {col}: {len(unseen)} value(s) in test not seen in train -> {unseen}")

    # Acceptance checks
    assert X_train.shape[1] == X_test.shape[1] == len(feature_names), "feature count mismatch"
    assert np.isnan(X_train).sum() == 0 and np.isnan(X_test).sum() == 0, "NaNs present"
    assert np.isfinite(X_train).all() and np.isfinite(X_test).all(), "non-finite values present"
    assert set(np.unique(y_train)) <= {0, 1} and set(np.unique(y_test)) <= {0, 1}, "labels not 0/1"
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test), "row/label mismatch"

    # Export
    save_matrix(X_train, feature_names, out_dir / 'X_train', args.format)
    save_matrix(X_test, feature_names, out_dir / 'X_test', args.format)
    y_train.to_csv(out_dir / 'y_train.csv', index=False)
    y_test.to_csv(out_dir / 'y_test.csv', index=False)
    joblib.dump(pre, out_dir / 'pipeline.pkl')
    with open(out_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)

    # Summary
    print(f"sklearn {sklearn.__version__} | format: {args.format}")
    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(f"numeric features: {len(NUMERICAL_COLS)}  one-hot features: {len(feature_names) - len(NUMERICAL_COLS)}")
    for name, y in (('train', y_train), ('test', y_test)):
        counts = y.value_counts().to_dict()
        print(f"y_{name}: normal={counts.get(0, 0)}  attack={counts.get(1, 0)}  "
              f"attack ratio={y.mean():.3f}")
    print(f"All checks passed. Files written to {out_dir.resolve()}")


if __name__ == '__main__':
    main()