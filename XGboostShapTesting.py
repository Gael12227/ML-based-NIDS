import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import DMatrix, XGBClassifier
from shap import TreeExplainer, summary_plot

data = pd.read_csv('data.csv')

X = data.drop('target', axis=1)  # Features
y = data['target']  # Target variable

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=100,  
    learning_rate=0.1,  
    max_depth=5, # to be test between 3 and 10
    subsample=0.8, # test between 0.5 and 1.0
    objective='binary:logistic',  
    tree_method='hist',
    device = 'cuda'  
)  # Initialize the XGBoost classifier

dxtrain = DMatrix(x_train, device='cuda:0')
dytrain = DMatrix(y_train, device='cuda:0')

trained_model = model.fit(dxtrain, dytrain)  # Train on data.csv

predictions = trained_model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
classification_rep = classification_report(y_test, predictions)
confusion_mat = confusion_matrix(y_test, predictions)

# Create a TreeExplainer for the trained model
explainer = TreeExplainer(trained_model)
summary = explainer.shap_values(x_test)
summary_plot(summary, x_test, feature_names=x_test.columns)
