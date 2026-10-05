import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

data = pd.read_csv('data.csv')  # Load your dataset here

X = data.drop('target', axis=1)  # Features
y = data['target']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split the dataset

model = XGBClassifier(
    early_stopping_rounds=50,  
    n_estimators=10000,  
    learning_rate=0.1,  
    max_depth=5, # to test between 3 and 10
    subsample=0.8, # test between 0.5 and 1.0
    objective='binary:logistic',  
    tree_method='gpu_hist',  
)  # Initialize the XGBoost classifier

score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

print("Cross-validation accuracy scores:", score)
print("Mean cross-validation accuracy:", score.mean(),"±", score.std())

predictions = model.fit(X_train, y_train).predict(X_test)  
accuracy = accuracy_score(y_test, predictions)
classification_rep = classification_report(y_test, predictions)
confusion_mat = confusion_matrix(y_test, predictions)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print("Test set accuracy:", accuracy)
print("Classification report:\n", classification_rep)
print("Confusion matrix:\n", confusion_mat)
print("ROC AUC score:", roc_auc)


print("Weights of the model:", model.get_booster().get_score(importance_type='weight'))

# model.save_model('xgboost_model.json') 
# model.load_model("my_xgboost_model.json") to load it again 