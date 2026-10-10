import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
import optuna
from optuna_integration.xgboost import XGBoostPruningCallback

# Load your dataset
data = pd.read_csv('data.csv')
X = data.drop('target', axis=1)
y = data['target']
def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',  # Required for pruning
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'tree_method': 'hist',
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'device': 'cuda',
        'n_estimators': 1000, # Set high, early stopping will handle the rest
        'early_stopping_rounds': 50,
    }
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        pruning_callback = XGBoostPruningCallback(trial, "validation_0-logloss")
        model = XGBClassifier(**param, callbacks=[pruning_callback])
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            verbose=False
        )
        scores.append(model.best_score) # Use the best score found during fitting
        
    return np.mean(scores)
# Run Optimization
study = optuna.create_study(direction="minimize",pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=50)

print("Best trial parameters:", study.best_params)

# Re-train using the best parameters found
best_model = XGBClassifier(**study.best_params, tree_method='hist', device='cuda')

# Perform Final CV
CV = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

print("Cross-validation scores:", CV)
print("Mean CV score:", CV.mean(), "+/-", CV.std())

"""
Best trial parameters: {'lambda': 2.9437887390310524, 'alpha': 0.012030941248539765, 'max_depth': 6, 'learning_rate': 0.25970398558545676, 'subsample': 0.8386394393978515, 'reg_lambda': 1.4440203352841714, 'reg_alpha': 0.039833020738182186}
Cross-validation scores: [0.99908712 0.99912681 0.99869022 0.99916647 0.99904739]
Mean CV score: 0.9990236016182612 +/- 0.0001713530924989805
"""