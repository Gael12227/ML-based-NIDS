import joblib
import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

data = pd.read_csv('data.csv')  # Load your dataset here

X = data.drop('target', axis=1)  # Features
y = data['target']  # Target variable

print("Code started")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split the dataset
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
score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

print(f"Cross-validation accuracy scores: {score}") 
print(f"Mean cross-validation accuracy: {score.mean():.3f} ± {score.std():.3f}")

trained_model = model.fit(X_train, y_train)

predictions = model.fit(X_train, y_train).predict(X_test)  

accuracy = accuracy_score(y_test, predictions)
classification_rep = classification_report(y_test, predictions)
confusion_mat = confusion_matrix(y_test, predictions)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

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
