from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
import joblib

def train_model(X_train, y_train):
    rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=500, random_state=42)
    chain = ClassifierChain(rf_model, order=[0, 1, 2, 3])
    chain.fit(X_train, y_train)
    return chain

def save_model(model, path: str):
    joblib.dump(model, path)