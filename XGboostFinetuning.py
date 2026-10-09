import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
import optuna

# Load your dataset
data = pd.read_csv('data.csv')
X = data.drop('target', axis=1)
y = data['target']

def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42,
    }
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        model = XGBClassifier(**param)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            verbose=False
        )
        scores.append(model.score(X.iloc[val_idx], y.iloc[val_idx]))
        
    return np.mean(scores)

# Run Optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best trial parameters:", study.best_params)

# Re-train using the best parameters found
best_model = XGBClassifier(**study.best_params, tree_method='hist', device='cuda')

# Perform Final CV
CV = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

print("Cross-validation scores:", CV)
print("Mean CV score:", CV.mean(), "+/-", CV.std())
