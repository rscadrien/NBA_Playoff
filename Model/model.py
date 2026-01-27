from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
import joblib
from xgboost import XGBClassifier

def train_model(X_train, y_train):
    #rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=500,max_depth= 8,min_samples_leaf = 20, max_features = 0.5)
    #xgb_model = XGBClassifier(eval_metric='logloss')
    base_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    chain = ClassifierChain(base_model, order=[0, 1, 2, 3])
    chain.fit(X_train, y_train)
    return chain

def save_model(model, path: str):
    joblib.dump(model, path)