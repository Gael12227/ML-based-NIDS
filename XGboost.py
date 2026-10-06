import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

data = pd.read_csv('data.csv')  # Load your dataset here

X = data.drop('target', axis=1)  # Features
y = data['target']  # Target variable

print("Code started")

data2= pd.read_csv('data2.csv')  # Load your dataset here

X_1= data2.drop('target', axis=1)  # Features
y_1= data2['target']  # Target variable


# Keep the external test set on the training schema. Categories that occur only
# in the test file are represented by zero columns because the model cannot
# learn parameters for categories absent from training.
X_1 = X_1.reindex(columns=X.columns, fill_value=0)

model = XGBClassifier(
    n_estimators=100,  
    learning_rate=0.1,  
    max_depth=5, # to test between 3 and 10
    subsample=0.8, # test between 0.5 and 1.0
    objective='binary:logistic',  
    tree_method='hist',
    device = 'cuda'  
)  # Initialize the XGBoost classifier

print("Starting Cross validation")
score = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Cross-validation accuracy scores: {score}") 
print(f"Mean cross-validation accuracy: {score.mean():.3f} ± {score.std():.3f}")

trained_model = model.fit(X, y)  # Train on data.csv

predictions = trained_model.predict(X_1)

accuracy = accuracy_score(y_1, predictions)
classification_rep = classification_report(y_1, predictions)
confusion_mat = confusion_matrix(y_1, predictions)
roc_auc = roc_auc_score(y_1, trained_model.predict_proba(X_1)[:, 1])

print(f"Test set accuracy: {accuracy:.3f}")
print(f"Classification report:\n{classification_rep}")
print(f"Confusion matrix:\n{confusion_mat}")
print(f"ROC AUC score: {roc_auc:.3f}")



trained_model.save_model('xgboost_model.json') 
# model.load_model("my_xgboost_model.json") to load it again 

with open('Report.txt', 'w') as f:

    f.write("XGBoost Model Report\n")

    f.write("Cross-validation accuracy scores: {}\n".format(score))
    f.write("Mean cross-validation accuracy: {:.3f} ± {:.3f}\n".format(score.mean(), score.std()))
    f.write("\n")

    f.write("Accuracy: {:.3f}\n".format(accuracy))
    f.write("Classification Report:\n{}\n".format(classification_rep))
    f.write("Confusion Matrix:\n{}\n".format(confusion_mat))
    f.write("ROC AUC Score: {:.3f}\n".format(roc_auc))
